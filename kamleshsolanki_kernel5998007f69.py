import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
PATH = "/kaggle/input/college-basketball-dataset"

os.listdir(PATH)
cbb_df = pd.read_csv(os.path.join(PATH,'cbb.csv'))

cbb15_df = pd.read_csv(os.path.join(PATH,'cbb15.csv'))

cbb16_df = pd.read_csv(os.path.join(PATH,'cbb16.csv'))

cbb17_df = pd.read_csv(os.path.join(PATH,'cbb17.csv'))

cbb18_df = pd.read_csv(os.path.join(PATH,'cbb18.csv'))

cbb19_df = pd.read_csv(os.path.join(PATH,'cbb19.csv'))
cbb_df.head()
cbb_df.shape
cbb_df.nunique()
cbb_df.describe()
cbb_df.sort_values(by='W',ascending=False).head(10).plot.bar(x='TEAM',y='W',alpha=0.36,color='red')
cbb_df.sort_values(by='ADJOE',ascending=False).head(10).plot.bar(x='TEAM',y='ADJOE',alpha=0.36,color='blue')
cbb_df.sort_values(by='ADJDE',ascending=False).head(10).plot.bar(x='TEAM',y='ADJDE',alpha=0.36,color='green')
cbb_df.sort_values(by='ADJDE',ascending=False).head(10).plot.bar(x='TEAM',y=['ADJDE','ADJOE','W'],alpha=0.36)