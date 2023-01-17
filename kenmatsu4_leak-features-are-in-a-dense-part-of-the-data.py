import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageDraw, ImageColor
train = pd.read_csv('../input/train.csv')

col = [c for c in train.columns if c not in ['ID', 'target']]
xtrain = train[col].copy().values

xtrain_round = (xtrain/1000).round()
xtrain_round_log1p = np.log1p(xtrain_round)
max_ = xtrain_round_log1p.max()
min_ = xtrain_round_log1p.min() 
field = np.round((xtrain_round_log1p - min_)/(max_ - min_)*255)
# rows and cols of leaked data
# from https://www.kaggle.com/johnfarrell/giba-s-property-extended-extended-result

rows = ['6726fff18',  'd94655f86',  '1df3ca92e',  'd8e48b069',  '24204cd10',  'bf6c2d1ef',  'eacc7ab9e',  '4a5425356',  'f0a57697d',  '625e88875',  'ca6cab485',  
        '2e8ee92a7',  '0e8c830c4',  '502cdea84',  '7862786dc',  'c95732596',  '16a02e67a',  'ad960f947',  '8adafbb52',  'fd0c7cfc2',  'a36b78ff7',  'e42aae1b8',  
        '0b132f2c6',  '448efbb28',  'ca98b17ca',  '2e57ec99f',  'fef33cb02',  'd4546ed8f',  '0e1920aa8',  '500d02a95',  'fae83c142',  'c5e859554',  'fba64a995',  
        '35dcc0108',  'b9713bb06',  '2514ad945']
cols =   ['f190486d6',  '58e2e02e6',  'eeb9cd3aa',  '9fd594eec',  '6eef030c1',  '15ace8c9f',  'fb0f5dbfe',  '58e056e12',  '20aa07010',  '024c577b9',  'd6bb78916', 
          'b43a7cfd5',  '58232a6fb',  '1702b5bf0',  '324921c7b',  '62e59a501',  '2ec5b290f',  '241f0f867',  'fb49e4212',  '66ace2992',  'f74e8f13d',  '5c6487af1',  '963a49cdc',  
          '26fc93eb7',  '1931ccfdd',  '703885424',  '70feb1494',  '491b9ee45',  '23310aa6f',  'e176a204a',  '6619d81fc',  '1db387535',  'fc99f9426',  '91f701ba2',  '0572565c2',  
          '190db8488',  'adb64ff71',  'c47340d97',  'c5a231d81']

# 
row_idx = train.ID.tolist()
col_idx = train.columns[2:].tolist()

# Counting none zero values for each row and column.
train_nz = (train != 0).astype(int)
train_nz = train_nz.iloc[:,2:]
sum_in_row = train_nz.sum(axis=1)
sum_in_col = train_nz.sum(axis=0)

#  getting sort key with argsort 
sum_in_row_argsort = np.argsort(sum_in_row).values[::-1]
sum_in_col_argsort = np.argsort(sum_in_col).values[::-1]

# sorting train data on row and column
xtrain_sort = xtrain[sum_in_row_argsort, :]
xtrain_sort = xtrain_sort[:,sum_in_col_argsort]

# sorting row index and column index
row_idx_sorted = np.asanyarray(row_idx)[sum_in_row_argsort]
col_idx_sorted = np.asanyarray(col_idx)[sum_in_col_argsort]

xtrain_sort_round = (xtrain_sort/1000).round()
xtrain_sort_round_log1p = np.log1p(xtrain_sort_round)
max_ = xtrain_sort_round_log1p.max()
min_ = xtrain_sort_round_log1p.min() 
field_argsort = np.round((xtrain_sort_round_log1p - min_)/(max_ - min_)*255)
WORK_SIZE = 0x100

COLOR_START = ImageColor.getrgb('white')
COLOR_STOP = ImageColor.getrgb('red')
gradation = np.array([np.linspace(i, j, WORK_SIZE) for i, j in zip(COLOR_START, COLOR_STOP)],dtype=int)

COLOR_START = ImageColor.getrgb('white')
COLOR_STOP = ImageColor.getrgb('blue')
gradation2 = np.array([np.linspace(i, j, WORK_SIZE) for i, j in zip(COLOR_START, COLOR_STOP)],dtype=int)

img = Image.new('RGBA', field_argsort.shape)
for x in range(len(sum_in_row)):
    print(f"\rx={x}/{sum_in_col.shape[0]}", end="")
    for y in range(len(sum_in_col)):
        row_check = row_idx_sorted[x] in rows
        col_check = col_idx_sorted[y] in cols
        if row_check and col_check:
            img.putpixel((x, y), tuple(gradation[:, int(field_argsort[x][y])]))
        else:
            img.putpixel((x, y), tuple(gradation2[:, int(field_argsort[x][y])]))

img.save("train_sorted.png")
np.save("row_idx_sorted",row_idx_sorted)
np.save("col_idx_sorted",col_idx_sorted)
train_sorted = train.set_index("ID").loc[row_idx_sorted[::-1], col_idx_sorted[::-1]]
train_sorted.to_csv("./train_sorted.csv")
np.log1p(train_sorted).iloc[:500,:500].replace(0,np.nan).to_csv("./train_sorted_small_log_nan.csv")
