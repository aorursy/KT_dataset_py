# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/Power-Networks-LCL.csv")

data.shape[:]
data.head()
data['LCLid'].unique()

data['Acorn'].unique()


data=data.set_index('DateTime')
data=data.drop('stdorToU',1)
df1=data.loc[data.LCLid=='MAC000002']

df2=data.loc[data.LCLid=='MAC000003']

df3=data.loc[data.LCLid=='MAC000004']

df4=data.loc[data.LCLid=='MAC000006']

df5=data.loc[data.LCLid=='MAC000007']

df6=data.loc[data.LCLid=='MAC000008']

df7=data.loc[data.LCLid=='MAC000009']

df8=data.loc[data.LCLid=='MAC000010']

df9=data.loc[data.LCLid=='MAC000011']

df10=data.loc[data.LCLid=='MAC000012']



df11=data.loc[data.LCLid=='MAC000013']

df12=data.loc[data.LCLid=='MAC000016']

df13=data.loc[data.LCLid=='MAC000018']

df14=data.loc[data.LCLid=='MAC000019']

df15=data.loc[data.LCLid=='MAC000020']

df16=data.loc[data.LCLid=='MAC000021']

df17=data.loc[data.LCLid=='MAC000022']

df18=data.loc[data.LCLid=='MAC000023']

df19=data.loc[data.LCLid=='MAC000024']

df20=data.loc[data.LCLid=='MAC000025']



df21=data.loc[data.LCLid=='MAC000026']

df22=data.loc[data.LCLid=='MAC000027']

df23=data.loc[data.LCLid=='MAC000028']

df24=data.loc[data.LCLid=='MAC000029']

df25=data.loc[data.LCLid=='MAC000030']

df26=data.loc[data.LCLid=='MAC000032']



df27=data.loc[data.LCLid=='MAC000033']

df28=data.loc[data.LCLid=='MAC000034']

df29=data.loc[data.LCLid=='MAC000035']

df30=data.loc[data.LCLid=='MAC000036']



df1=df1.drop('LCLid',1)

df2=df2.drop('LCLid',1)

df3=df3.drop('LCLid',1)

df4=df4.drop('LCLid',1)

df5=df5.drop('LCLid',1)

df6=df6.drop('LCLid',1)

df7=df7.drop('LCLid',1)

df8=df8.drop('LCLid',1)

df9=df9.drop('LCLid',1)

df10=df10.drop('LCLid',1)

df11=df11.drop('LCLid',1)

df12=df12.drop('LCLid',1)

df13=df13.drop('LCLid',1)

df14=df14.drop('LCLid',1)

df15=df15.drop('LCLid',1)

df16=df16.drop('LCLid',1)

df17=df17.drop('LCLid',1)

df18=df18.drop('LCLid',1)

df19=df19.drop('LCLid',1)

df20=df20.drop('LCLid',1)

df21=df21.drop('LCLid',1)

df22=df22.drop('LCLid',1)

df23=df23.drop('LCLid',1)

df24=df24.drop('LCLid',1)

df25=df25.drop('LCLid',1)

df26=df26.drop('LCLid',1)

df27=df27.drop('LCLid',1)

df28=df28.drop('LCLid',1)

df29=df29.drop('LCLid',1)

df30=df30.drop('LCLid',1)

df1.index = df1.index.astype("datetime64")

df2.index = df2.index.astype("datetime64")

df3.index = df3.index.astype("datetime64")

df4.index = df4.index.astype("datetime64")

df5.index = df5.index.astype("datetime64")

df6.index = df6.index.astype("datetime64")

df7.index = df7.index.astype("datetime64")

df8.index = df8.index.astype("datetime64")

df9.index = df9.index.astype("datetime64")

df10.index = df10.index.astype("datetime64")

df11.index = df11.index.astype("datetime64")

df12.index = df12.index.astype("datetime64")

df13.index = df13.index.astype("datetime64")

df14.index = df14.index.astype("datetime64")

df15.index = df15.index.astype("datetime64")

df16.index = df16.index.astype("datetime64")

df17.index = df17.index.astype("datetime64")

df18.index = df18.index.astype("datetime64")

df19.index = df19.index.astype("datetime64")

df20.index = df20.index.astype("datetime64")

df21.index = df21.index.astype("datetime64")

df22.index = df22.index.astype("datetime64")

df23.index = df23.index.astype("datetime64")

df24.index = df24.index.astype("datetime64")

df25.index = df25.index.astype("datetime64")

df26.index = df26.index.astype("datetime64")

df27.index = df27.index.astype("datetime64")

df28.index = df28.index.astype("datetime64")

df29.index = df29.index.astype("datetime64")

df30.index = df30.index.astype("datetime64")

df1.head()
df1['Acorn'] = df1['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df2['Acorn'] = df2['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df3['Acorn'] = df3['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df4['Acorn'] = df4['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df5['Acorn'] = df5['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df6['Acorn'] = df6['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df7['Acorn'] = df7['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df8['Acorn'] = df8['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df9['Acorn'] = df9['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df10['Acorn'] = df10['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df11['Acorn'] = df11['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df12['Acorn'] = df12['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df13['Acorn'] = df13['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df14['Acorn'] = df14['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df15['Acorn'] = df15['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df16['Acorn'] = df16['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df17['Acorn'] = df17['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df18['Acorn'] = df18['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df19['Acorn'] = df19['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df20['Acorn'] = df20['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df21['Acorn'] = df21['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df22['Acorn'] = df22['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df23['Acorn'] = df23['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df24['Acorn'] = df24['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df25['Acorn'] = df25['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df26['Acorn'] = df26['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df27['Acorn'] = df27['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df28['Acorn'] = df28['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df29['Acorn'] = df29['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df30['Acorn'] = df30['Acorn'].apply(lambda x : x.strip().split('-')[-1]) 

df2.head()