!pip install /kaggle/input/efficientnet-pytorch/efficientnet_pytorch-0.6.1-py3-none-any.whl

!pip install /kaggle/input/ptmodels/pretrainedmodels-0.7.4-py3-none-any.whl
import gc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



import torch

import torch.utils.data as data_utils

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torch import nn

import cv2



# from torchsummary import summary

from torchvision import transforms,models

import torch.nn.functional as F

from tqdm import tqdm

from torch import Tensor



# from efficientnet_pytorch import EfficientNet

from collections import OrderedDict



import albumentations

from PIL import Image

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.augmentations import functional as Fun

from PIL import Image, ImageOps, ImageEnhance

import pretrainedmodels



from albumentations.core.transforms_interface import DualTransform



from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

class ResNet34(nn.Module):

    def __init__(self, pretrained):

        super(ResNet34, self).__init__()

        if pretrained is True:

            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")

        else:

            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        

        self.l0 = nn.Linear(512, 168)

        self.l1 = nn.Linear(512, 11)

        self.l2 = nn.Linear(512, 7)



    def forward(self, x):

        bs, _, _, _ = x.shape

        x = self.model.features(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)

        l1 = self.l1(x)

        l2 = self.l2(x)

        return l0, l1, l2
class EfficientNetWrapper(nn.Module):

    def __init__(self, pretrained):

        super(EfficientNetWrapper, self).__init__()

        

        # Load imagenet pre-trained model 

        self.effNet = EfficientNet.from_name('efficientnet-b3')

        

        # Appdend output layers based on our date

        self.fc_root = nn.Linear(in_features=1000, out_features=168)

        self.fc_vowel = nn.Linear(in_features=1000, out_features=11)

        self.fc_consonant = nn.Linear(in_features=1000, out_features=7)

        

    def forward(self, X):

        output = self.effNet(X)

        output_root = self.fc_root(output)

        output_vowel = self.fc_vowel(output)

        output_consonant = self.fc_consonant(output)

        

        return output_root, output_vowel, output_consonant
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_MEAN = (0.485, 0.456, 0.406)

MODEL_STD = (0.229, 0.224, 0.225)

IMG_HEIGHT = 137

IMG_WIDTH = 236
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

    img_ids_list = []

    FOLDS = 5

    TEST_BATCH_SIZE = 12



    final_img_ids = []

    

    ffinal_g_pred = []

    ffinal_v_pred = []

    ffinal_c_pred = []



    for file_idx in range(4):

        final_g_pred = []

        final_v_pred = []

        final_c_pred = []

        df = pd.read_parquet(f"/kaggle/input/bengaliai-cv19/test_image_data_{file_idx}.parquet")



        dataset = BengaliDatasetTest(df=df,

                                    img_height=IMG_HEIGHT,

                                    img_width=IMG_WIDTH,

                                    mean=MODEL_MEAN,

                                    std=MODEL_STD)

        

        del df

        gc.collect()

        torch.cuda.empty_cache()



        data_loader = torch.utils.data.DataLoader(

            dataset=dataset,

            batch_size= TEST_BATCH_SIZE,

            shuffle=False

        )

        

        for fold in range(FOLDS):

            

            model1 = EfficientNetWrapper(False)

            model2 = ResNet34(False)

            model3 = EfficientNetWrapper(False)

            

            g_pred1, v_pred1, c_pred1 = [], [], []

            g_pred2, v_pred2, c_pred2 = [], [], []

            g_pred3, v_pred3, c_pred3 = [], [], []

            

            model1.load_state_dict(torch.load(f"/kaggle/input/bengaliaieffnetb3/efficientNet_fold{fold}.pth", map_location=DEVICE))

            model1.to(DEVICE)

            model1.eval()

            

            model2.load_state_dict(torch.load(f"/kaggle/input/resnet34originalsize/resnet34_fold{fold}.pth", map_location=DEVICE))

            model2.to(DEVICE)

            model2.eval()

            

            model3.load_state_dict(torch.load(f"/kaggle/input/effnetb3-cutmix-mixup/efficientNet_fold{fold}.pth", map_location=DEVICE))

            model3.to(DEVICE)

            model3.eval()



            for bi, d in tqdm(enumerate(data_loader)):

                image = d["image"]

                img_id = d["image_id"]

                image = image.to(DEVICE, dtype=torch.float32)

                

                g1, v1, c1 = model1(image)

                g2, v2, c2 = model2(image)

                g3, v3, c3 = model3(image)



                for ii, imid in enumerate(img_id):

                    g_pred1.append(g1[ii].cpu().detach().numpy())

                    v_pred1.append(v1[ii].cpu().detach().numpy())

                    c_pred1.append(c1[ii].cpu().detach().numpy())

                    

                    g_pred2.append(g2[ii].cpu().detach().numpy())

                    v_pred2.append(v2[ii].cpu().detach().numpy())

                    c_pred2.append(c2[ii].cpu().detach().numpy())

                    

                    g_pred3.append(g3[ii].cpu().detach().numpy())

                    v_pred3.append(v3[ii].cpu().detach().numpy())

                    c_pred3.append(c3[ii].cpu().detach().numpy())

                    if fold == 0:

                        final_img_ids.append(imid)



            final_g_pred.append(g_pred1)

            final_v_pred.append(v_pred1)

            final_c_pred.append(c_pred1)

            

            final_g_pred.append(g_pred2)

            final_v_pred.append(v_pred2)

            final_c_pred.append(c_pred2)

            

            final_g_pred.append(g_pred3)

            final_v_pred.append(v_pred3)

            final_c_pred.append(c_pred3)

            

            del g_pred1

            del g_pred2

            del g_pred3

            del c_pred1

            del c_pred2

            del c_pred3

            del v_pred1

            del v_pred2

            del v_pred3

            gc.collect()

            torch.cuda.empty_cache()

             

        ffinal_g_pred.append(final_g_pred)

        ffinal_v_pred.append(final_v_pred)

        ffinal_c_pred.append(final_c_pred)

        

        del final_g_pred

        del final_v_pred

        del final_c_pred

        del data_loader

        del dataset

        gc.collect()

        torch.cuda.empty_cache()

      

    del model1

    del model2

    del model3

    gc.collect()

    torch.cuda.empty_cache()

    return ffinal_g_pred, ffinal_v_pred, ffinal_c_pred, final_img_ids
final_g_pred, final_v_pred, final_c_pred, final_img_ids = model_predict()
final_g = np.argmax(np.mean(np.array(final_g_pred), axis=1).reshape(-1, 168), axis=1)

final_v = np.argmax(np.mean(np.array(final_v_pred), axis=1).reshape(-1, 11), axis=1)

final_c = np.argmax(np.mean(np.array(final_c_pred), axis=1).reshape(-1, 7), axis=1)
del final_g_pred

del final_v_pred

del final_c_pred

gc.collect()
predictions = []

for ii, imid in enumerate(final_img_ids):

    predictions.append((f"{imid}_grapheme_root", final_g[ii]))

    predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))

    predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))
del final_g

del final_v

del final_c

gc.collect()
sub = pd.DataFrame(predictions, columns=["row_id", "target"])

sub.to_csv("submission.csv", index=False)

sub