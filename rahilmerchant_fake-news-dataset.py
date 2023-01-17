import numpy as np 

import pandas as pd
data_path1 = '/kaggle/input/fake-news-cc/'

data_cc = pd.read_csv(data_path1 + 'news.csv')

data_path2 = '/kaggle/input/fake-and-real-news-dataset/'

data2_fake=pd.read_csv(data_path2 + 'Fake.csv')

data2_real=pd.read_csv(data_path2 + 'True.csv')
data_cc.head()
data2_fake.head()
data2_real.head(80)
data2_real['label']=1

data2_fake['label']=0
data2=pd.concat([data2_real,data2_fake])
print(data2.shape)

data2.head()
data_cc.drop('Unnamed: 0', inplace = True, axis = 1)
data2.drop(['subject','date'],inplace=True,axis=1)
data2.head()
data_cc=data_cc.replace(to_replace ="FAKE", value =0) 

data_cc=data_cc.replace(to_replace ="REAL", value =1) 
data_cc.tail()
data_cc.head()
data2.head()
data=pd.concat([data_cc,data2])

print(data.shape)

data.head()
data = data.dropna()

data = data.reset_index(drop=True)

print(data.shape)

data.head()
data['label'].value_counts()
data.to_csv('fake-news-data.csv', index=False)