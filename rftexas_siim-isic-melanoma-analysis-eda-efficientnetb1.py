import matplotlib.pyplot as plt

import seaborn as sns



import cv2

from PIL import Image



import numpy as np

import pandas as pd



import glob
DATA_ROOT_PATH = "../input/siim-isic-melanoma-classification/"

TRAIN_CSV = DATA_ROOT_PATH + "train.csv"

TEST_CSV = DATA_ROOT_PATH + "test.csv"



train_df = pd.read_csv(TRAIN_CSV, na_values=['unknown'])

test_df = pd.read_csv(TEST_CSV)
trn_len_df = len(train_df)

tst_len_df = len(test_df)

print(f"There are {trn_len_df} images in the training set")

print(f"There are {tst_len_df} images in the test set")
train_df.head()
test_df.head()
# NaN values



nan_stats = train_df.isna().sum() / len(train_df) * 100



stats = pd.DataFrame({

    'columns': train_df.columns,

    'NaN statistics (in %)': nan_stats

})



stats = stats.sort_values(by=['NaN statistics (in %)'], ascending=False)



stats = stats.reset_index(drop=True)

stats.head(5)
# Patient distribution

print(f"There are {train_df['patient_id'].nunique()} unique patients for {len(train_df)} images in the training set.")

print(f"There are {test_df['patient_id'].nunique()} unique patients for {len(test_df)} images in the training set.")



fig, ax = plt.subplots(1, 2, figsize=(15, 10))



sns.countplot(train_df['patient_id'], ax=ax[0])

ax[0].set_title('Patient distribution in the training set')



sns.countplot(test_df['patient_id'], ax=ax[1])

ax[1].set_title('Patient distribution in the test set')



plt.show()
trn_patients = set(train_df['patient_id'])

tst_patients = set(test_df['patient_id'])



inter_patients = len(trn_patients.intersection(tst_patients))



print(f'There are {inter_patients} common patients in the training and test sets.')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))



sns.countplot(train_df['sex'], ax=ax[0])

ax[0].set_title("Sex distribution in the training set")



sns.countplot(test_df['sex'], ax=ax[1])

ax[1].set_title("Sex distribution in the test set")



plt.show()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))



sns.distplot(train_df['age_approx'], ax=ax[0])

ax[0].set_title("Age distribution in the training set")



sns.distplot(test_df['age_approx'], ax=ax[1])

ax[1].set_title("Age distribution in the test set")



plt.show()
fig, ax = plt.subplots(1, 2, figsize=(10, 5))



chart = sns.countplot(train_df['anatom_site_general_challenge'], ax=ax[0])

ax[0].set_title("Anatomical site in the training set")

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)



chart2 = sns.countplot(test_df['anatom_site_general_challenge'], ax=ax[1])

ax[1].set_title("Anatomical site in the test set")

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=45)



plt.show()
chart3 = sns.countplot(train_df['diagnosis'])

chart3.set_xticklabels(chart3.get_xticklabels(), rotation=45)



plt.show()
print(f"There are {len(train_df[train_df['target'] == 0])} negative labels.")

print(f"There are {len(train_df[train_df['target'] == 1])} positive labels.")



sns.countplot(train_df['target'])

plt.show()
fig = plt.figure(figsize=(7,5))

ax = sns.countplot(x="target", hue="sex", data=train_df)



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2, height+10, '{:1.2f}%'.format(100*height/len(train_df)), ha="center")
fig = plt.figure(figsize=(7,5))

ax = sns.countplot(x="target", hue="anatom_site_general_challenge", data=train_df)



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2, height+15, '{:1.2f}%'.format(100*height/len(train_df)), ha="center")
# Training set



img_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/train/*.jpg')



fig, ax = plt.subplots(4, 4, figsize=(20, 20))



for i in range(16):

    x = i // 4

    y = i % 4

    

    path = img_names[i]

    image_id = path.split("/")[5][:-4]

    

    target = train_df.loc[train_df['image_name'] == image_id, 'target'].tolist()[0]

    

    img = Image.open(path)

    

    ax[x, y].imshow(img)

    ax[x, y].axis('off')

    ax[x, y].set_title(f'ID: {image_id}, Target: {target}')



fig.suptitle("Training set samples", fontsize=15)
# Test set



img_names = glob.glob('../input/siim-isic-melanoma-classification/jpeg/test/*.jpg')



fig, ax = plt.subplots(4, 4, figsize=(20, 20))



for i in range(16):

    x = i // 4

    y = i % 4

    

    path = img_names[i]

    image_id = path.split("/")[5][:-4]



    img = Image.open(path)

    

    ax[x, y].imshow(img)

    ax[x, y].axis('off')

    ax[x, y].set_title(f'ID: {image_id}')



fig.suptitle("Test set samples", fontsize=15)
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
!pip install -q efficientnet_pytorch
import os, re, random, gc

from tqdm import tqdm



import numpy as np

import pandas as pd



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.optim import Adam, lr_scheduler

from torch.utils.data.sampler import WeightedRandomSampler



#import pretrainedmodels

from efficientnet_pytorch import EfficientNet



import albumentations

from PIL import Image



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
WIDTH = 224

HEIGHT = 224

TRAIN_BATCH_SIZE = 128

VALID_BATCH_SIZE = 128

EPOCHS = 10

LR = 1e-3

FOLDS = 5

SEED = 0

VERBOSE_STEP = 1

TRAIN_CSV = '/kaggle/input/siim-isic-melanoma-classification/train.csv'

SUBMISSION_CSV = '/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv'



MEAN = (0.485, 0.456, 0.406)

STD = (0.229, 0.224, 0.225)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True



seed_everything(SEED)
class AverageMeter:

    def __init__(self):

        self.reset()

    

    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0

    

    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count
class MelanomaDataset:

    def __init__(self, image_paths, targets, resize=True, augmentations=None):

        self.image_paths = image_paths

        self.targets = targets

        self.augmentations = augmentations

        self.resize = resize

    

    def __len__(self):

        return len(self.image_paths)

    

    def __getitem__(self, item):

        image = Image.open(self.image_paths[item])

        targets = self.targets[item]

        

        if self.resize:

            image = image.resize(

                (WIDTH, HEIGHT), resample=Image.BILINEAR

            )

        

        image = np.array(image)

        

        if self.augmentations is not None:

            augmented = self.augmentations(image=image)

            image = augmented['image']

        

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        

        return {

            'image': torch.tensor(image, dtype=torch.float),

            'targets': torch.tensor(targets, dtype=torch.long),

        }
df = pd.read_csv(TRAIN_CSV)

df['image_name'] = df['image_name'].apply(lambda x: f'../input/siic-isic-224x224-images/train/{x}.png')

df.head()
# Generating folds



kf = KFold(FOLDS, random_state=SEED)

df = df.sample(frac=1).reset_index(drop=True)



for f, (_, val_index) in enumerate(kf.split(df, df)):

    df.loc[val_index, 'kfold'] = f

    

print(df['kfold'].value_counts())
# Deriving weights for sampler

def generate_weights(df):

    B = 0.5



    C = np.array([B, (1 - B)])*2

    ones = len(df.query('target == 1'))

    zeros = len(df.query('target == 0'))



    weightage_fn = {0: C[1]/zeros, 1: C[0]/ones}

    return [weightage_fn[target] for target in df.target]
train_transforms = albumentations.Compose([

    albumentations.ShiftScaleRotate(p=0.9),

    albumentations.CLAHE(p=0.5),

    albumentations.HorizontalFlip(p=0.5),

    albumentations.VerticalFlip(p=0.5),

    albumentations.RandomBrightnessContrast(p=0.9),

    albumentations.Normalize(mean=MEAN, std=STD, always_apply=True)

])



test_transforms = albumentations.Compose([

    albumentations.Normalize(mean=MEAN, std=STD, always_apply=True)

])
class MelanomaModel(nn.Module):

    def __init__(self):

        super(MelanomaModel, self).__init__()

        

        self.encoder = EfficientNet.from_pretrained("efficientnet-b1")

        self.dropout = nn.Dropout(0.3)

        self.head = nn.Linear(1280, 1)

    

    def forward(self, image):

        batch_size, _, _, _ = image.shape

        

        x = self.encoder.extract_features(image)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        

        x = self.dropout(x)

        logit = self.head(x)

        

        return logit
class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):

        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.logits = logits

        self.reduce = reduce



    def forward(self, inputs, targets):

        if self.logits:

            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=None)

        else:

            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=None)

        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss



        if self.reduce:

            return torch.mean(F_loss)

        else:

            return F_loss
def loss_fn(outputs, targets):

    return FocalLoss(logits=True)(outputs, targets)
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    

    losses = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):

        images = d['image']

        targets = d['targets']

        

        images = images.to(device, dtype=torch.float)

        targets = targets.to(device, dtype=torch.long)



        model.zero_grad()

        outputs = model(images)

        targets = targets.view(-1, 1).type_as(outputs)

        

        loss = loss_fn(outputs, targets)

        loss.backward()

        xm.optimizer_step(optimizer, barrier=True)

        

        if scheduler:

            scheduler.step()

        

        losses.update(loss.item(), images.size(0))

        

        tk0.set_postfix(loss=losses.avg)
def eval_fn(data_loader, model, device):

    model.eval()

    

    losses = AverageMeter()

    final_preds = []

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        

        for bi, d in enumerate(tk0):

            images = d['image']

            targets = d['targets']



            images = images.to(device, dtype=torch.float)

            targets = targets.to(device, dtype=torch.long)



            outputs = model(images)

            targets = targets.view(-1, 1).type_as(outputs)

            

            loss = loss_fn(outputs, targets)

            losses.update(loss.item(), images.size(0))

            

            final_preds.extend(outputs.cpu().detach().numpy().tolist())

        

    return losses.avg, final_preds
def run_fold(fold):

    device = xm.xla_device()

    model = MelanomaModel().to(device)

    best_auc = 0

    

    # Selecting fold

    train_df = df[df['kfold'] != fold].reset_index(drop=True)

    valid_df = df[df['kfold'] == fold].reset_index(drop=True)

    

    weights = generate_weights(train_df)

        

    # Loading data

    

    train_dataset = MelanomaDataset(

        image_paths=train_df['image_name'],

        targets=train_df['target'],

        resize=True,

        augmentations=train_transforms,

    )



    valid_dataset = MelanomaDataset(

        image_paths=valid_df['image_name'],

        targets=valid_df['target'],

        resize=True,

        augmentations=test_transforms,

     )



    train_sampler = WeightedRandomSampler(weights, len(train_df))



    train_loader = torch.utils.data.DataLoader(

        train_dataset, 

        batch_size=TRAIN_BATCH_SIZE, 

        sampler=train_sampler,

        num_workers=8

    )



    valid_loader = torch.utils.data.DataLoader(

        valid_dataset, 

        batch_size=VALID_BATCH_SIZE, 

        shuffle=False, 

        num_workers=8

    )

    

    # Optimizer and scheduler

    

    num_train_steps = int(len(train_df) / TRAIN_BATCH_SIZE * EPOCHS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=num_train_steps,

        eta_min=1e-6

    )

    

    # Training loop

    

    for epoch in range(EPOCHS):

        train_fn(train_loader, model, optimizer, device=device, scheduler=scheduler)

        loss, y_pred = eval_fn(valid_loader, model, device=device)

        

        y_pred = np.array(y_pred)

        val_auc = roc_auc_score(valid_df['target'].values, y_pred)

        

        xm.master_print(f"Epoch = {epoch}, val_loss = {loss}, val_auc = {val_auc}")

        

        if val_auc > best_auc:

            xm.save(model.state_dict(), f"model_{fold}.bin")

            xm.master_print('Validation score improved ({} --> {}). Saving model!'.format(best_auc, val_auc))

            best_auc = val_auc
run_fold(0)
run_fold(1)
run_fold(2)
run_fold(3)
run_fold(4)
class MelanomaDataset:

    def __init__(self, image_paths, resize=True, augmentations=None):

        self.image_paths = image_paths

        self.augmentations = augmentations

        self.resize = resize

    

    def __len__(self):

        return len(self.image_paths)

    

    def __getitem__(self, item):

        image = Image.open(self.image_paths[item])

        

        if self.resize:

            image = image.resize(

                (WIDTH, HEIGHT), resample=Image.BILINEAR

            )

        

        image = np.array(image)

        

        if self.augmentations is not None:

            augmented = self.augmentations(image=image)

            image = augmented['image']

        

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        

        return {

            'image': torch.tensor(image, dtype=torch.float),

        }
test_df = pd.read_csv(TEST_CSV)

test_df['image_name'] = test_df['image_name'].apply(lambda x: f'../input/siic-isic-224x224-images/test/{x}.png')

test_df.head()
test_aug_transforms = albumentations.Compose([

    

    albumentations.ShiftScaleRotate(p=0.9),

    

    albumentations.OneOf([

        

        albumentations.CLAHE(p=0.5),

        albumentations.HueSaturationValue(p=0.5),

        

    ]),

    

    albumentations.OneOf([

    

        albumentations.HorizontalFlip(p=0.5),

        albumentations.VerticalFlip(p=0.5),

        

    ]),

    

    albumentations.RandomBrightnessContrast(p=0.9),

    

    albumentations.Normalize(mean=MEAN, std=STD, always_apply=True)

    

])
test_transforms = albumentations.Compose([

    albumentations.Normalize(mean=MEAN, std=STD, always_apply=True)

])
def predict(data_loader, model, device):

    model.eval()

    

    final_preds = []

    

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))

        

        for bi, d in enumerate(tk0):

            images = d['image']

            

            images = images.to(device, dtype=torch.float)



            outputs = model(images)

                        

            final_preds.extend(outputs.cpu().detach().numpy().tolist())

        

    return final_preds
test_dataset = MelanomaDataset(

    image_paths=test_df['image_name'],

    resize=True,

    augmentations=test_transforms,

)



test_loader = torch.utils.data.DataLoader(

    test_dataset, 

    batch_size=VALID_BATCH_SIZE, 

    shuffle=False, 

    num_workers=8

)





test_aug_dataset = MelanomaDataset(

    image_paths=test_df['image_name'],

    resize=True,

    augmentations=test_aug_transforms,

)



test_aug_loader = torch.utils.data.DataLoader(

    test_aug_dataset,

    batch_size=VALID_BATCH_SIZE,

    shuffle=False,

    num_workers=8

)
MODEL_PATHS = [f'model_{fold}.bin' for fold in range(5)]
device = xm.xla_device()

predictions = []



for path in MODEL_PATHS:

    model = MelanomaModel().to(device)

    model.load_state_dict(torch.load(path))

    

    preds = predict(test_loader, model, device)

    preds_aug = predict(test_aug_loader, model, device)

    

    preds = np.array(preds)

    preds_aug = np.array(preds_aug)

    

    final_preds = np.mean([preds, preds_aug], axis=0)

    

    predictions.append(final_preds)
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
predictions = np.array(predictions)

predictions = np.mean(predictions, axis=0)
predictions = sigmoid(predictions)
print(predictions)
sub_df = pd.read_csv(SUBMISSION_CSV)

sub_df['target'] = predictions
sub_df.to_csv("submission.csv", index=False)