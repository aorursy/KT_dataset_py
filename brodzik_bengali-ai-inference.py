!pip install /kaggle/input/efficientnet-pytorch

!pip install /kaggle/input/pytorchcv/pytorchcv-0.0.57-py2.py3-none-any.whl
import glob

import os

import zipfile



import cv2

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from efficientnet_pytorch import EfficientNet

from pytorchcv.model_provider import get_model

from torch.utils.data import DataLoader, Dataset
INPUT_FILES = ["/kaggle/input/bengaliai-cv19/test_image_data_0.parquet",

               "/kaggle/input/bengaliai-cv19/test_image_data_1.parquet",

               "/kaggle/input/bengaliai-cv19/test_image_data_2.parquet",

               "/kaggle/input/bengaliai-cv19/test_image_data_3.parquet"]



HEIGHT = 137

WIDTH = 236



BATCH_SIZE = 50



DEVICE = torch.device("cuda")

DTYPE = torch.float
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax
def crop_resize(img0, size, pad=16):

    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax, xmin:xmax]

    img[img < 28] = 0

    lx, ly = xmax-xmin, ymax-ymin

    l = max(lx, ly) + pad

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode="constant")

    return cv2.resize(img, (size, size))
def preprocess_images(input_files, output_file, size):

    x_tot, x2_tot = [], []



    with zipfile.ZipFile(output_file, "w") as img_out:

        for fname in input_files:

            df = pd.read_parquet(fname)

            data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

            for idx in range(len(df)):

                name = df.iloc[idx, 0]



                img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)

                img = crop_resize(img, size)



                x_tot.append((img/255.0).mean())

                x2_tot.append(((img/255.0)**2).mean())

                img = cv2.imencode(".png", img)[1]

                img_out.writestr(name + ".png", img)
class MyDataset(Dataset):

    def __init__(self, img_dir, img_files):

        self.img_dir = img_dir

        self.img_files = img_files



    def __len__(self):

        return len(self.img_files)



    def __getitem__(self, idx):

        img = cv2.imread(self.img_files[idx])

        img = img.transpose((2, 0, 1))[0]

        img = img / 255

        return int(self.img_files[idx].replace(self.img_dir, "").replace(".png", "").replace("Test_", "")), np.array([img])
preds = {}
if not os.path.isfile("test_images_128.zip"):

    preprocess_images(INPUT_FILES, "test_images_128.zip", 128)



with zipfile.ZipFile("test_images_128.zip") as img_zip:

    img_zip.extractall("test_images_128")



test_set = MyDataset("test_images_128/", glob.glob("test_images_128/*.png"))

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
##################################################

# EfficientNet B5 Epoch 28 (Public score: 0.9638)

##################################################

model = EfficientNet.from_name_2("efficientnet-b5", num_classes=186, in_channels=1)

model.load_state_dict(torch.load("/kaggle/input/bengaliai-trained-models/model_b5_28.pth"))

model = model.to(device=DEVICE, dtype=DTYPE)

model = model.eval()



with torch.no_grad():

    for test_id, X in test_loader:

        test_id = test_id.numpy()

        X = X.to(device=DEVICE, dtype=DTYPE)



        y_pred = model(X)

        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy() * 0.9638



        for i, pred in list(zip(test_id, y_pred)):

            if i in preds:

                preds[i] += pred

            else:

                preds[i] = pred
##################################################

# EfficientNet B2 epoch 20 (Public score: 0.9629)

##################################################

model = EfficientNet.from_name_2("efficientnet-b2", num_classes=186, in_channels=1)

model.load_state_dict(torch.load("/kaggle/input/bengaliai-trained-models/model_b2_20.pth"))

model = model.to(device=DEVICE, dtype=DTYPE)

model = model.eval()



with torch.no_grad():

    for test_id, X in test_loader:

        test_id = test_id.numpy()

        X = X.to(device=DEVICE, dtype=DTYPE)



        y_pred = model(X)

        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy() * 0.9629



        for i, pred in list(zip(test_id, y_pred)):

            if i in preds:

                preds[i] += pred

            else:

                preds[i] = pred
!ls
!rm -rf test_images_128 test_images_128.zip
!ls
if not os.path.isfile("test_images_224.zip"):

    preprocess_images(INPUT_FILES, "test_images_224.zip", 224)



with zipfile.ZipFile("test_images_224.zip") as img_zip:

    img_zip.extractall("test_images_224")



test_set = MyDataset("test_images_224/", glob.glob("test_images_224/*.png"))

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
##################################################

# seresnext50_32x4d epoch 16 (Public score: 0.9630)

##################################################

model = get_model("seresnext50_32x4d", pretrained=False)

model.features.init_block.conv.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.output = nn.Linear(in_features=2048, out_features=186, bias=True)

model.load_state_dict(torch.load("/kaggle/input/bengaliai-trained-models/model_seresnext50_32x4d_16.pth"))

model = model.to(device=DEVICE, dtype=DTYPE)

model = model.eval()



with torch.no_grad():

    for test_id, X in test_loader:

        test_id = test_id.numpy()

        X = X.to(device=DEVICE, dtype=DTYPE)



        y_pred = model(X)

        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy() * 0.9630



        for i, pred in list(zip(test_id, y_pred)):

            if i in preds:

                preds[i] += pred

            else:

                preds[i] = pred
!ls
!rm -rf test_images_224 test_images_224.zip
!ls
submission = []
for i, pred in preds.items():

    y1 = np.argmax(pred[:168]) # grapheme_root

    y2 = np.argmax(pred[168:179]) # vowel_diacritic

    y3 = np.argmax(pred[179:]) # consonant_diacritic



    submission.append((i, "Test_" + str(i) + "_grapheme_root", y1))

    submission.append((i, "Test_" + str(i) + "_vowel_diacritic", y2))

    submission.append((i, "Test_" + str(i) + "_consonant_diacritic", y3))
submission = pd.DataFrame(submission, columns=["id", "row_id", "target"]).sort_values(by=["id", "row_id"]).reset_index()[["row_id", "target"]]

submission.to_csv("submission.csv", index=False)
!ls