import os

import PIL

import cv2

import torch

import fastai

import numpy as np

import pandas as pd

from fastai.vision import *

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from torchvision import models, transforms

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, cohen_kappa_score



torch.manual_seed(0)

torch.backends.cudnn.deterministic = True



print(os.listdir("../input"))
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def load_ben_color(image, image_size,sigmaX=10):

    image = np.array(image)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, image_size)

    image = cv2.addWeighted(image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    image = PIL.Image.fromarray(image)

    return image



class BenColor:

    def __call__(self, img, image_size=(512, 512)):

        return load_ben_color(img, image_size)
class TrainDataset(Dataset):

    def __init__(self, dataframe, transforms):

        self.data = dataframe

        self.transform = transform

        self.c = 5

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img_name = os.path.join(

            "../input/aptos2019-blindness-detection/train_images/{}".format(self.data.loc[idx, "id_code"])

        )

        image = PIL.Image.open(img_name)

        image = self.transform(image)

        label = torch.tensor(self.data.loc[idx, "diagnosis"])

        return image, label





class TestDataset(Dataset):

    def __init__(self, df, transform):

        self.data = df

        self.transform = transform

        self.c = 5

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        img_path = os.path.join("../input/aptos2019-blindness-detection/test_images/", self.data.loc[idx, "id_code"])

        img = PIL.Image.open(img_path)

        img = self.transform(img)

        return img

IMG_SIZE = 512



train_table = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test_table = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")





train_table["id_code"] = train_table["id_code"].apply(lambda x: x + ".png")

test_table["id_code"] = test_table["id_code"].apply(lambda x: x + ".png")



x_train, x_val, y_train, y_val = train_test_split(train_table["id_code"], train_table["diagnosis"], random_state=0)



x_train.index = np.arange(len(x_train))

y_train.index = np.arange(len(y_train))

x_val.index = np.arange(len(x_val))

y_val.index = np.arange(len(y_val))





transform = transforms.Compose([

    BenColor(),

    transforms.RandomChoice([

        transforms.RandomCrop(299),

        transforms.Resize(299)

    ]),

    transforms.RandomChoice([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip()

    ]),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



val_transform = transforms.Compose([

    BenColor(),

    transforms.Resize(299),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

train_ds = TrainDataset(pd.concat([x_train, y_train], axis=1), transform)

val_ds = TrainDataset(pd.concat([x_val, y_val], axis=1), val_transform)

test_ds = TestDataset(test_table, val_transform)



train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)

val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)

test_loader = DataLoader(test_ds, batch_size=32, num_workers=4)
body = create_body(models.resnet50, False, None)

head = create_head(4096, 5)



model = nn.Sequential(body, head)

model.load_state_dict(

    torch.load("../input/pretraind-fastai-resnet50/fastai_resnet50_epochs.pth")

)
class Extraction:

    def __init__(self, network):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.network = network.eval().to(self.device)

        

    def for_train(self, loader):

        data_tmp = []

        label_tmp = []

        

        with torch.no_grad():

            for x, y in loader:

                x = x.to(self.device)

            

                outputs = self.network[0](x)

                outputs = self.network[1][0](outputs)

                data_tmp.append(outputs.view(-1, 4096).cpu().numpy())

                

                label_tmp.append(y.cpu().numpy())

                

        return np.vstack(data_tmp), np.hstack(label_tmp)

        

    

    def for_test(self, loader):

        tmp = []

        with torch.no_grad():

            for x in loader:

                x = x.to(self.device)

            

                outputs = self.network[0](x)

                outputs = self.network[1][0](outputs)

                tmp.append(outputs.view(-1, 4096).cpu().numpy())

        return np.vstack(tmp)
ext = Extraction(model)



train_feature, train_label = ext.for_train(train_loader)

val_feature, val_label = ext.for_train(val_loader)

test_feature = ext.for_test(test_loader)
train_feature.shape
pca = PCA()

pca.fit(train_feature)
print(np.round(pca.explained_variance_ratio_, 3)[:4])

print("-"*50)

print(np.round(pca.explained_variance_ratio_, 5)[:4].sum())
train_fea = pca.transform(train_feature)[:, :4]

val_fea = pca.transform(val_feature)[:, :4]

test_fea = pca.transform(test_feature)[:, :4]
pd.DataFrame(train_fea[:, :2]).plot(kind="scatter", x=0, y=1, c=train_label, cmap="tab10", figsize=(15, 10))

plt.show()
torch.cuda.empty_cache()



tree = XGBClassifier(random_state=1997)

tree.fit(train_fea, train_label)
cohen_kappa_score(val_label, tree.predict(val_fea), weights="quadratic")
submit = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

submit["diagnosis"] = tree.predict(test_fea)

submit.to_csv("submission.csv", index=False)