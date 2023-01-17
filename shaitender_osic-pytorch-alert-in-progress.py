!pip install wtfml
!pip install pretrainedmodels
import os
import torch
import albumentations

import numpy as np
import pandas as pd

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels
class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()
        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss
# create folds
df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
df['Patient_Week'] = df['Patient'].astype(str) + '_' + df['Weeks'].astype(str)
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df['FVC'].values
kf = model_selection.StratifiedKFold(n_splits=3)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv("train_folds.csv", index=False)
import glob
import pydicom

train_image_path = '../input/osic-pulmonary-fibrosis-progression/train'
train_image_files = glob.glob(os.path.join(train_image_path, '*', '*.dcm'))

train_image_data = pydicom.read_file(train_image_files[0])
train_image_data
import matplotlib.pyplot as plt
%matplotlib inline

filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
df_folds = pd.read_csv("train_folds.csv")
print(df_folds['kfold'].value_counts())

df_folds.head()

#todo - changes

def train(fold):
    training_data_path = "../input/osic-pulmonary-fibrosis-progression/train/"
    df = pd.read_csv("/kaggle/working/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = SEResnext50_32x4d(pretrained="imagenet")
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_Patient_Weeks = df_train.Patient_Week.values.tolist()
    train_Patient_Weeks = [os.path.join(training_data_path, i) for i in train_Patient_Weeks]
    train_targets = df_train['FVC'].values

    valid_Patient_Weeks = df_valid.Patient_Week.values.tolist()
    valid_Patient_Weeks = [os.path.join(training_data_path, i) for i in valid_Patient_Weeks]
    valid_targets = df_train['FVC'].values

    train_dataset = ClassificationLoader(
        image_paths=train_Patient_Weeks,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_Patient_Weeks,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break
def predict(fold):
    test_data_path = "../input/osic-pulmonary-fibrosis-progression/test/"
    df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
    device = "cuda"
    model_path=f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    patient_Weeks = df.Patient_Week.values.tolist()
    patient_Weeks = [os.path.join(test_data_path, i) for i in patient_Weeks]
    targets = np.zeros(len(patient_Weeks))

    test_dataset = ClassificationLoader(
        image_paths=patient_Weeks,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions
train(0)
train(1)
train(2)
p1 = predict(0)
p2 = predict(1)
p3 = predict(2)
predictions = (p1 + p2 + p3) / 3
sample = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sample.head()