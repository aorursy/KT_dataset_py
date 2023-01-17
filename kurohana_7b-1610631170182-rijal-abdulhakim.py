import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_excel('../input/Data.xls')

df.head() #untuk menampilkan 5 record awal dari dataset sebagai penguji dataset sudah termuat atau belum
df.describe()
df.info()
#membuat dataframe baru

df_outlier = df.drop(['ID','Reason for absence','Disciplinary failure'],axis=1)
#menghitung IQR

IQR = df_outlier.quantile(0.75) - df_outlier.quantile(0.25)

IQR
#menghitung batas atas & batas atas ekstrem

upper_limit = df_outlier.quantile(0.75) + (IQR * 1.5)

upper_limit_extreme = df_outlier.quantile(0.75) + (IQR * 3)

print("Batas Atas :\n",upper_limit)

print("\nBatas Atas Ekstrem :\n",upper_limit_extreme)
#menghitung batas bawah & batas bawah ekstrem

lower_limit = df_outlier.quantile(0.75) - (IQR * 1.5)

lower_limit_extreme = df_outlier.quantile(0.75) - (IQR * 3)

print("Batas Bawah :\n",lower_limit)

print("\nBatas Bawah Ekstrem :\n",lower_limit_extreme)
for col in df_outlier.columns:

    outlier=0

    for i in range(0,df_outlier.shape[0]):

        if(df_outlier[col][i] > upper_limit[col]):

            outlier+=1

        elif(df_outlier[col][i] < lower_limit[col]):

            outlier+=1

    print("Outlier",col,":",outlier,"(",(float)(outlier/df_outlier.shape[0]),"%)")
df_outlier=df_outlier.drop(['Month of absence','Day of the week','Seasons','Distance from Residence to Work','Age'],axis=1)

cols = df_outlier.columns



#Top Coding

for col in cols:

    df.loc[df[col] > upper_limit[col], col] = upper_limit[col]



#Bottom Coding

for col in cols:

    df.loc[df[col] < lower_limit[col], col] = lower_limit[col]



#Zero Coding

for col in cols:

    df.loc[df[col] < lower_limit[col], col] = 0



df.describe()
df.loc[df['Age'] > 55, 'Age'] = 4

df.loc[df['Age'] > 45, 'Age'] = 3

df.loc[df['Age'] > 35, 'Age'] = 2

df.loc[df['Age'] > 26, 'Age'] = 1

df.describe()
df.corr()
df=df.drop(['ID'], axis=1)

df