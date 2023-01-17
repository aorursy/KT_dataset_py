import os

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/stage_2_train.csv')

len(df)
ndf = df.drop_duplicates(['ImageId'])

ndf['EncodedPixels'] = ' ' + ndf.loc[:,['EncodedPixels']]

ndf = ndf.rename(columns={"EncodedPixels":" EncodedPixels"})

len(ndf)
ndf.to_csv('train-rle.csv', index=False)