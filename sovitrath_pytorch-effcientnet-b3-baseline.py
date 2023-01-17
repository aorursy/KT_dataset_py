# install pytorh efficientnet

!pip install efficientnet_pytorch
# set the path for iertstrat for creating cross-validation CSV file

import sys

iterstrat = '../input/iterative-stratification/iterative-stratification-master/'

sys.path.insert(0, iterstrat)
import os

import pandas as pd

import joblib

import albumentations

import numpy as np

import torch

import torchvision.transforms as transforms

import cv2

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

import matplotlib.style as style



from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from PIL import Image



style.use('ggplot')
df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')

print(df.head)

# create a kfold col

df.loc[:, 'kfold'] = -1



# shuffle the dataframe (reset index keeps the index same as before)

df = df.sample(frac=1).reset_index(drop=True)

print(df.head)



X = df.image_id.values

# we have multiple label columns

y = df[['healthy', 'multiple_diseases', 'rust', 'scab']].values



mskf = MultilabelStratifiedKFold(n_splits=5)



for fold, (trn_, val_) in enumerate(mskf.split(X, y)):

    # trn_ and val_ are index values in the df

    print('TRAIN: ', trn_, 'VAL: ', val_)

    df.loc[val_, 'kfold'] = fold



print(df.kfold.value_counts())

df.to_csv('train_folds.csv', index=False)
class PlantDatasetTrain:

    def __init__(self, folds, img_height, img_width, mean, std):

        df = pd.read_csv('train_folds.csv')

        # only grab the cols that we need

        df = df[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab', 'kfold']]



        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        self.image_ids = df.image_id.values

        self.healthy = df.healthy.values

        self.multiple_diseases = df.multiple_diseases.values

        self.rust = df.rust.values

        self.scab = df.scab.values



        # apply augmentations

        if len(folds) == 1: # if validating

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.Normalize(mean, std, always_apply=True)

            ])

        else: # if training

            self.aug = albumentations.Compose([

                albumentations.Resize(img_height, img_width, always_apply=True),

                albumentations.ShiftScaleRotate(

                    shift_limit=0.0625,

                    scale_limit=0.1,

                    rotate_limit=5,

                    p=0.9

                ),

                albumentations.Normalize(mean, std, always_apply=True)

            ])

        

    def __len__(self):

        return len(self.image_ids)



    def __getitem__(self, item):

        image = cv2.imread(f"../input/resized/train_224/{self.image_ids[item]}.jpg")

        image = cv2.resize(image, (224, 224))

        image = self.aug(image=np.array(image))['image']

        # from (h, w, c) to (c, h, w)

        image = np.transpose(image, (2, 0, 1)).astype(np.float)

        return {

            'image': torch.tensor(image, dtype=torch.float),

            'healthy': torch.tensor(self.healthy[item], dtype=torch.long),

            'multiple_diseases': torch.tensor(self.multiple_diseases[item], dtype=torch.long),

            'rust': torch.tensor(self.rust[item], dtype=torch.long),

            'scab': torch.tensor(self.scab[item], dtype=torch.long)

        }
class EfficientNetB3(nn.Module):

    def __init__(self, pretrained):

        super(EfficientNetB3, self).__init__()

        if pretrained == True:

            self.model = EfficientNet.from_pretrained('efficientnet-b3')

        else:

            self.model = EfficientNet.from_name('efficientnet-b3')

        

        self.l0 = nn.Linear(1536, 2)

        self.l1 = nn.Linear(1536, 2)

        self.l2 = nn.Linear(1536, 2)

        self.l3 = nn.Linear(1536, 2)



    def forward(self, x):

        # get the batch size only, ignore (c, h, w)

        bs, _, _, _ = x.shape

        x = self.model.extract_features(x)

        # reshape makes the number of rows equal to batch_size

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)

        l3 = self.l3(x)

        return l0, l1, l2, l3
MODELS = {

    'efficientnet-b3': EfficientNetB3,

}
def loss_fn(outputs, targets):

    o1, o2, o3, o4 = outputs

    t1, t2, t3, t4 = targets

    l1 = nn.CrossEntropyLoss()(o1, t1)

    l2 = nn.CrossEntropyLoss()(o2, t2)

    l3 = nn.CrossEntropyLoss()(o3, t3)

    l4 = nn.CrossEntropyLoss()(o4, t4)

    return (l1 + l2 + l3 + l4) / 4



def train(dataset, data_loader, model, optimizer):

    model.train()

    final_loss = 0

    counter = 0

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):

        counter = counter + 1

        image = d['image']

        healthy = d['healthy']

        multiple_diseases = d['multiple_diseases']

        rust = d['rust']

        scab = d['scab']



        image = image.to(DEVICE, dtype=torch.float)

        healthy = healthy.to(DEVICE, dtype=torch.long)

        multiple_diseases = multiple_diseases.to(DEVICE, dtype=torch.long)

        rust = rust.to(DEVICE, dtype=torch.long)

        scab = scab.to(DEVICE, dtype=torch.long)



        optimizer.zero_grad()

        outputs = model(image)

        targets = (healthy, multiple_diseases, rust, scab)

        loss = loss_fn(outputs, targets)

        final_loss += loss



        loss.backward()

        optimizer.step()

    print(f"Train loss: {(final_loss/counter):.3f}")

    return final_loss/counter



def evaluate(dataset, data_loader, model):

    model.eval()

    final_loss = 0

    counter = 0

    with torch.no_grad():

        for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset)/data_loader.batch_size)):

            counter = counter + 1

            image = d['image']

            healthy = d['healthy']

            multiple_diseases = d['multiple_diseases']

            rust = d['rust']

            scab = d['scab']



            image = image.to(DEVICE, dtype=torch.float)

            healthy = healthy.to(DEVICE, dtype=torch.long)

            multiple_diseases = multiple_diseases.to(DEVICE, dtype=torch.long)

            rust = rust.to(DEVICE, dtype=torch.long)

            scab = scab.to(DEVICE, dtype=torch.long)



            outputs = model(image)

            targets = (healthy, multiple_diseases, rust, scab)

            loss = loss_fn(outputs, targets)

            final_loss += loss

    print(f"Val loss: {(final_loss/counter):.3f}")

    return final_loss / counter



def main():

    model = MODELS[BASE_MODEL](pretrained=True)

    model.to(DEVICE)

    

    # Find total parameters and trainable parameters

#     total_params = sum(p.numel() for p in model.parameters())

#     print(f'{total_params:,} total parameters.')

#     total_trainable_params = sum(

#         p.numel() for p in model.parameters() if p.requires_grad)

#     print(f'{total_trainable_params:,} training parameters.')



    train_dataset = PlantDatasetTrain(

        folds=TRAINING_FOLDS,

        img_height=IMG_HEIGHT,

        img_width=IMG_WIDTH,

        mean=MODEL_MEAN,

        std=MODEL_STD

    )



    train_loader = torch.utils.data.DataLoader(

        dataset=train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        shuffle=True,

        num_workers=4,

    )



    valid_dataset = PlantDatasetTrain(

        folds=VALIDATION_FOLDS,

        img_height=IMG_HEIGHT,

        img_width=IMG_WIDTH,

        mean=MODEL_MEAN,

        std=MODEL_STD,

    )



    valid_loader = torch.utils.data.DataLoader(

        dataset=valid_dataset,

        batch_size=TEST_BATCH_SIZE,

        shuffle=False,

        num_workers=4,

    )



    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 

        optimizer,

        mode='min',

        patience=5,

        factor=0.5,

        verbose=True

    )



    # train

    train_loss, val_loss = [], []

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch+1} of {EPOCHS}")

        train_score = train(train_dataset, train_loader, model, optimizer)

        val_score = evaluate(valid_dataset, valid_loader, model)

        train_loss.append(train_score)

        val_loss.append(val_score)

        # run scheduler based on val_score

        scheduler.step(val_score)

        torch.save(model.state_dict(), f"{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")



    # loss plots

    plt.figure(figsize=(10, 7))

    plt.plot(train_loss, color='orange', label='train loss')

    plt.plot(val_loss, color='red', label='validataion loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.savefig(f"loss_{BASE_MODEL}_{VALIDATION_FOLDS}.png")

    plt.show()
if torch.cuda.is_available():

    DEVICE = 'cuda'

else:

    DEVICE = 'cpu'

    

# parameters / options

IMG_HEIGHT=224

IMG_WIDTH=224

EPOCHS=30

TRAIN_BATCH_SIZE=32

TEST_BATCH_SIZE=16

MODEL_MEAN=[0.485, 0.456, 0.406]

MODEL_STD=[0.229, 0.224, 0.225]

BASE_MODEL='efficientnet-b3'

TRAINING_FOLDS_CSV='../input/train_folds.csv'
TRAINING_FOLDS=(0, 1, 2, 3)

VALIDATION_FOLDS=(4,)

main()
TRAINING_FOLDS=(0, 1, 2, 4)

VALIDATION_FOLDS=(3,)

main()
TRAINING_FOLDS=(0, 1, 4, 3)

VALIDATION_FOLDS=(2,)

main()
TRAINING_FOLDS=(0, 4, 2, 3)

VALIDATION_FOLDS=(1,)

main()
TRAINING_FOLDS=(4, 1, 2, 3)

VALIDATION_FOLDS=(0,)

main()
import glob 

import albumentations

import torch

import pandas as pd

import joblib

import torch.nn as nn

import numpy as np

import cv2 as cv



from tqdm import tqdm

from PIL import Image

from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet



TEST_BATCH_SIZE = 96

MODEL_MEAN = (0.485, 0.456, 0.406)

MODEL_STD = (0.229, 0.224, 0.225)

IMG_HEIGHT = 224

IMG_WIDTH = 224



if torch.cuda.is_available():

    DEVICE = 'cuda'

else:

    DEVICE = 'cpu'



###################################################################################################

class EfficientNetB3(nn.Module):

    def __init__(self, pretrained):

        super(EfficientNetB3, self).__init__()

        if pretrained == True:

            self.model = EfficientNet.from_pretrained('efficientnet-b3')

        else:

            self.model = EfficientNet.from_name('efficientnet-b3')

        

        # number of classes per label (3 different linear layers)

        self.l0 = nn.Linear(1536, 2)

        self.l1 = nn.Linear(1536, 2)

        self.l2 = nn.Linear(1536, 2)

        self.l3 = nn.Linear(1536, 2)



    def forward(self, x):

        # get the batch size only, ignore (c, h, w)

        bs, _, _, _ = x.shape

        x = self.model.extract_features(x)

        # reshape makes the number of rows equal to batch_size

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)

        l3 = self.l3(x)

        return l0, l1, l2, l3

##########################################################################



class PlantDatasetTest:

    def __init__(self, df, img_height, img_width, mean, std):

        self.image_ids = df.image_id.values

        self.aug = albumentations.Compose([

            albumentations.Resize(img_height, img_width, always_apply=True),

            albumentations.Normalize(mean, std, always_apply=True)

        ])



    def __len__(self):

        return len(self.image_ids)



    def __getitem__(self, item):

        image = cv.imread(f"../input/plant-pathology-2020-fgvc7/images/{self.image_ids[item]}.jpg")

        img_id = self.image_ids[item]

        image = cv.resize(image, (224, 224))

        image = self.aug(image=np.array(image))['image']

        # from (h, w, c) to (c, h, w)

        image = np.transpose(image, (2, 0, 1)).astype(np.float)

        return {

            'image': torch.tensor(image, dtype=torch.float),

            'image_id': img_id

        }



''' Models '''

models = []

model4 = EfficientNetB3(pretrained=False).to(DEVICE)

model4.load_state_dict(torch.load(f"efficientnet-b3_fold4.bin"))

models.append(model4)



model3 = EfficientNetB3(pretrained=False).to(DEVICE)

model3.load_state_dict(torch.load(f"efficientnet-b3_fold3.bin"))

models.append(model3)



model2 = EfficientNetB3(pretrained=False).to(DEVICE)

model2.load_state_dict(torch.load(f"efficientnet-b3_fold2.bin"))

models.append(model2)



model1 = EfficientNetB3(pretrained=False).to(DEVICE)

model1.load_state_dict(torch.load(f"efficientnet-b3_fold1.bin"))

models.append(model1)



model0 = EfficientNetB3(pretrained=False).to(DEVICE)

model0.load_state_dict(torch.load(f"efficientnet-b3_fold0.bin"))

models.append(model0)

''' Models '''



def model_predict(i):

    with torch.no_grad():

        h_pred, m_pred, r_pred, s_pred = [], [], [], []

        img_ids_list = [] 

        

        df = pd.read_csv(f"../input/plant-pathology-2020-fgvc7/test.csv")



        dataset = PlantDatasetTest(df=df,

                                    img_height=IMG_HEIGHT,

                                    img_width=IMG_WIDTH,

                                    mean=MODEL_MEAN,

                                    std=MODEL_STD)



        data_loader = torch.utils.data.DataLoader(

            dataset=dataset,

            batch_size=TEST_BATCH_SIZE,

            shuffle=False,

            # num_workers=4

        )



        for bi, d in enumerate(data_loader):

            image = d["image"]

            img_id = d["image_id"]

            image = image.to(DEVICE, dtype=torch.float)



            h, m, r, s = models[i](image)



            for ii, imid in enumerate(img_id):

                h_pred.append(h[ii].cpu().detach().numpy())

                m_pred.append(m[ii].cpu().detach().numpy())

                r_pred.append(r[ii].cpu().detach().numpy())

                s_pred.append(s[ii].cpu().detach().numpy())

                img_ids_list.append(imid)

        

    return h_pred, m_pred, r_pred, s_pred, img_ids_list



final_h_pred = []

final_m_pred = []

final_r_pred = []

final_s_pred = []

final_img_ids = []



for i in range(len(models)):

    # model.eval()

    h_pred, m_pred, r_pred, s_pred, img_ids_list = model_predict(i)

    

    final_h_pred.append(h_pred)

    final_m_pred.append(m_pred)

    final_r_pred.append(r_pred)

    final_s_pred.append(s_pred)

    if i == 0:

        final_img_ids.extend(img_ids_list)



final_h = (np.mean(np.array(final_h_pred), axis=0))

final_m = (np.mean(np.array(final_m_pred), axis=0))

final_r = (np.mean(np.array(final_r_pred), axis=0))

final_s = (np.mean(np.array(final_s_pred), axis=0))



predictions = []

for ii, imid in enumerate(final_img_ids):

    predictions.append((f"{imid}", final_h[ii][1], final_m[ii][1], final_r[ii][1], final_s[ii][1]))



sub = pd.DataFrame(predictions, columns=['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab'])



sub.to_csv('submission.csv', index=False)



print('Successfully created submission file')