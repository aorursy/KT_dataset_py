import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

path = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/Metadata_Indicator.csv')

df2 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/Metadata_Country.csv')

df3 = pd.read_csv('/kaggle/input/ncd-who-dataset/NCD_WHO_Data/WHO-cause-of-death-by-NCD.csv')
df1.head()
# SOURCE NOTE

print(df1['SOURCE_NOTE'].unique())
df2.head()
df3.head()
df3.info()
# Drop years from 1960-1999 (no data)

df3 = df3.drop(df3.iloc[:,4:44],axis=1)
df3.info()
# Drop years from 2017 onwards (no data)

df3 = df3.drop(df3.iloc[:,21:25],axis=1)
df3.head()
print(df3['Indicator Name'].unique())
print(df3['Indicator Code'].unique())
df3 = df3.drop(['Indicator Name','Indicator Code'],axis=1)
df3.head()