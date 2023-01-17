

import pandas as pd

import joblib

import glob

from tqdm import tqdm

import pandas as pd

import albumentations

import joblib

import numpy as np

import torch



from PIL import Image

import torch.nn as nn

from torch.nn import functional as F
%%writefile train.py



import os

import ast

import torch

import torch.nn as nn

import numpy as np

import sklearn.metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys

from torchvision import models
import glob

import torch

import albumentations

import pandas as pd

import numpy as np



from tqdm import tqdm

from PIL import Image

import joblib

import torch.nn as nn

from torch.nn import functional as F
model_dir =  "../input/bengalifold6/fold6.pth"
class BengaliModel(nn.Module):

    def __init__(self, pretrained=False):

        super(BengaliModel, self).__init__()

        self.basemodel = models.resnext50_32x4d(pretrained=pretrained)

        

        self.Linr = nn.Linear(2048, 168)

        self.Linv = nn.Linear(2048, 11)

        self.Linc = nn.Linear(2048, 7)



    def forward(self, x):

        bs, _, _ , _ = x.shape



        x = self.basemodel.conv1(x)

        x = self.basemodel.bn1(x)

        x = self.basemodel.relu(x)

        x = self.basemodel.maxpool(x)

        x = self.basemodel.layer1(x)

        x = self.basemodel.layer2(x)

        x = self.basemodel.layer3(x)

        x = self.basemodel.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)



        roots = self.Linr(x)

        vowels = self.Linv(x)

        consonents = self.Linc(x)





        return roots, vowels, consonents





MODEL_MEAN = (0.485, 0.456, 0.406)

MODEL_STD = (0.229, 0.224, 0.225)

IMG_HEIGHT = 137

IMG_WIDTH = 236

DEVICE="cuda"
class BengaliDatasetTest:

    def __init__(self, df, img_height, img_width, mean, std):

        

        self.image_ids = df.image_id.values

        self.img_arr = df.iloc[:, 1:].values



        self.aug = albumentations.Compose([

            albumentations.Resize(img_height, img_width, always_apply=True),

            albumentations.Normalize(mean, std, always_apply=True)

        ])





    def __len__(self):

        return len(self.image_ids)

    

    def __getitem__(self, item):

        image = self.img_arr[item, :]

        img_id = self.image_ids[item]

        

        image = image.reshape(137, 236).astype(float)

        image = Image.fromarray(image).convert("RGB")

        image = self.aug(image=np.array(image))["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        



        return {

            "image": torch.tensor(image, dtype=torch.float),

            "image_id": img_id

        }
def model_predict():

    g_pred, v_pred, c_pred = [], [], []

    img_ids_list = [] 

    

    for file_idx in range(4):

        print('loading Parquet'+str(file_idx))

        df = pd.read_parquet(f"../input/bengaliai-cv19/test_image_data_{file_idx}.parquet")



        dataset = BengaliDatasetTest(df=df,

                                    img_height=IMG_HEIGHT,

                                    img_width=IMG_WIDTH,

                                    mean=MODEL_MEAN,

                                    std=MODEL_STD)



        data_loader = torch.utils.data.DataLoader(

            dataset=dataset,

            batch_size= TEST_BATCH_SIZE,

            shuffle=False,

            num_workers=4

        )



        for bi, d in enumerate(data_loader):

            image = d["image"]

            img_id = d["image_id"]

            image = image.to(DEVICE, dtype=torch.float)



            g, v, c = model(image)

            #g = np.argmax(g.cpu().detach().numpy(), axis=1)

            #v = np.argmax(v.cpu().detach().numpy(), axis=1)

            #c = np.argmax(c.cpu().detach().numpy(), axis=1)



            for ii, imid in enumerate(img_id):

                g_pred.append(g[ii].cpu().detach().numpy())

                v_pred.append(v[ii].cpu().detach().numpy())

                c_pred.append(c[ii].cpu().detach().numpy())

                img_ids_list.append(imid)

        

    return g_pred, v_pred, c_pred, img_ids_list
model = BengaliModel(pretrained=False)

model.load_state_dict(torch.load(model_dir))



TEST_BATCH_SIZE = 32



final_g_pred = []

final_v_pred = []

final_c_pred = []

final_img_ids = []



model.to(DEVICE)

model.eval()

g_pred, v_pred, c_pred, img_ids_list = model_predict()



final_g_pred.append(g_pred)

final_v_pred.append(v_pred)

final_c_pred.append(c_pred)

final_img_ids.extend(img_ids_list)
img_ids_list
torch.argmax(torch.softmax(torch.tensor(final_g_pred[0]), 0), 1).shape
final_g = np.argmax(np.array(final_g_pred[0]), axis=1)

final_v = np.argmax(np.array(final_v_pred[0]), axis=1)

final_c = np.argmax(np.array(final_c_pred[0]), axis=1)
final_img_ids
final_g.shape
predictions = []

for ii, imid in enumerate(final_img_ids):

    predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))

    predictions.append((f"{imid}_grapheme_root", final_g[ii]))

    predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))

    
sub = pd.DataFrame(predictions, columns=["row_id", "target"])
sub
sub.to_csv("submission.csv", index=False)