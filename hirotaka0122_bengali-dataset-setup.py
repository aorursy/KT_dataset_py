import gc

import joblib

import numpy as np

import pandas as pd

import pyarrow.parquet as pq

import matplotlib.pyplot as plt

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
H = 137

W = 236
%%time

PATH = "/kaggle/input/bengaliai-cv19/"

train_df = pd.read_csv(PATH+"train.csv")



data0 = pq.read_table(PATH+"train_image_data_0.parquet").to_pandas()

data0_id = data0["image_id"]

data0 = data0.iloc[:, 1:].astype(np.uint8).values

data0 = data0.reshape(-1, H, W, 1)                    # [N, H, W, C]



data1 = pq.read_table(PATH+"train_image_data_1.parquet").to_pandas()

data1_id = data1["image_id"]

data1 = data1.iloc[:, 1:].astype(np.uint8).values

data1 = data1.reshape(-1, H, W, 1)



data2 = pq.read_table(PATH+"train_image_data_2.parquet").to_pandas()

data2_id = data2["image_id"]

data2 = data2.iloc[:, 1:].astype(np.uint8).values

data2 = data2.reshape(-1, H, W, 1)



data3 = pq.read_table(PATH+"train_image_data_3.parquet").to_pandas()

data3_id = data3["image_id"]

data3 = data3.iloc[:, 1:].astype(np.uint8).values

data3 = data3.reshape(-1, H, W, 1)
print(train_df.shape)

train_df.head()
data_full = np.vstack([data0, data1, data2, data3])



del data0, data1, data2, data3

gc.collect()



id_full = pd.concat([data0_id, data1_id, data2_id, data3_id], ignore_index=True)

del data0_id, data1_id, data2_id, data3_id

gc.collect()

print(data_full.shape)

print(id_full.shape)
%%time

with open("train_data_full.joblib", "wb") as f:

    joblib.dump(data_full, f, compress=3)
class BengaliDataset(Dataset):

    def __init__(self, data: np.ndarray, label: pd.DataFrame, train: bool = True, transform: object = None):

        self.data: np.ndarray = data

        self.label: pd.DataFrame = label

        self.isTrain: bool = train

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, item):

        img = self.data[item]

        pil = transforms.ToPILImage()

        

        if self.transform is not None:

            img = pil(img)

            img = self.transform(img)

        

        if self.isTrain:

            label1 = self.label.grapheme_root.values[item]

            label2 = self.label.vowel_diacritic.values[item]

            label3 = self.label.consonant_diacritic.values[item]



            return img, label1, label2, label3

        else:

            return img
ds = BengaliDataset(

        data_full, 

        train_df, 

        train=True, 

        transform=transforms.Compose([

            transforms.CenterCrop(H), transforms.ToTensor()

        ])

    )



loader = DataLoader(ds, batch_size=2, shuffle=False)



x, y1, y2, y3 = next(iter(loader))
x.shape
plt.figure(facecolor="azure")

plt.imshow(x[0].permute(1, 2, 0).squeeze().numpy(), cmap="gray")



print("grapheme_root: ", y1[0])

print("vowel_diacritic: ", y2[0])

print("consonant_diacritic: ", y3[0])
del data_full, id_full, train_df, ds, loader, x, y1, y2, y3

gc.collect()
%%time

test_df = pd.read_csv(PATH+"test.csv")



data0 = pq.read_table(PATH+"test_image_data_0.parquet").to_pandas()

data0_id = data0["image_id"]

data0 = data0.iloc[:, 1:].astype(np.uint8).values

data0 = data0.reshape(-1, 137, 236, 1)



data1 = pq.read_table(PATH+"test_image_data_1.parquet").to_pandas()

data1_id = data1["image_id"]

data1 = data1.iloc[:, 1:].astype(np.uint8).values

data1 = data1.reshape(-1, 137, 236, 1)



data2 = pq.read_table(PATH+"test_image_data_2.parquet").to_pandas()

data2_id = data2["image_id"]

data2 = data2.iloc[:, 1:].astype(np.uint8).values

data2 = data2.reshape(-1, 137, 236, 1)



data3 = pq.read_table(PATH+"test_image_data_3.parquet").to_pandas()

data3_id = data3["image_id"]

data3 = data3.iloc[:, 1:].astype(np.uint8).values

data3 = data3.reshape(-1, 137, 236, 1)
data_full = np.vstack([data0, data1, data2, data3])



del data0, data1, data2, data3

gc.collect()



id_full = pd.concat([data0_id, data1_id, data2_id, data3_id], ignore_index=True)

del data0_id, data1_id, data2_id, data3_id

gc.collect()

print(data_full.shape)

print(id_full.shape)
%%time

with open("test_data_full.joblib", "wb") as f:

    joblib.dump(data_full, f, compress=3)
ds = BengaliDataset(

        data_full, 

        test_df, 

        train=False, 

        transform=transforms.Compose([

            transforms.CenterCrop(H), transforms.ToTensor()

        ])

    )

loader = DataLoader(ds, batch_size=2, shuffle=False)



x = next(iter(loader))
plt.figure(facecolor="azure")

plt.imshow(x[0].permute(1, 2, 0).squeeze().numpy(), cmap="gray")