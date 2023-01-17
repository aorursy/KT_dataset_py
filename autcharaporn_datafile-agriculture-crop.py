import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/agricuture-crops-production-in-india/datafile.csv')

df.head()
df.info()
df.dtypes
df.shape
#Display columns that contain null values.



df.isnull().any()
#List row indices that contain null values.



df[df.isnull().any(axis=1)==True].index
#ชุดข้อมูลมีค่าสูญหาย (missing value) จำนวนกี่แห่ง



row_nan = df[df.isnull().any(axis=1)==True].index

len(row_nan)
#แก้ปัญหาโดยใช้ drop row ค่า NAN



df.drop(row_nan,inplace=True)
df1 = df.copy()

df1.shape
df1.isnull().any()
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)

plt.show()