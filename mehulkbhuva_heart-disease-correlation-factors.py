import os

import warnings

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

import numpy as np

import math as mt

import scipy



from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
txtfile = "../input/cardiovascular-disease/cardiovascular.txt"

data = pd.read_csv(txtfile,sep=';',decimal=',')



data.index = data.iloc[:,0]

df = data.iloc[:,1:]



df = df.drop(['chd','famhist'],axis=1)

df.head()
df = df.astype('float')
df.dtypes
df.describe()
df=pd.DataFrame(df,columns=['sbp', 'tobacco', 'ldl', 'adiposity','obesity','alcohol', 'age','typea'])

df.index=data.index



corr = df.corr()

cmap = sea.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio

sea.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()