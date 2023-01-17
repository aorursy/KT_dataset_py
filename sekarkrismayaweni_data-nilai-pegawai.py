1. # This Python 3 environment comes with many helpful analytics libraries installed

1. # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

1. # For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



1. # Input data files are available in the "../input/" directory.

1. # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



 # Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df='../input/Data Nilai Pegawai.csv'

nilai_pegawai= pd.read_csv(df, index_col=0)

nilai_pegawai
sns.swarmplot(x='Tingkat Pendidikan Akhir', y='Nilai', data=nilai_pegawai)
sns.boxplot(x='Satuan Kerja', y='Nilai', data=nilai_pegawai, notch=True)
sns.regplot(x='Usia', y='Nilai', data=nilai_pegawai)
#Categorical Variabel

from sklearn.preprocessing import LabelEncoder



#Make copy to avoid changing original data

label_nilai_pegawai=nilai_pegawai.copy()



#List Categorical

object_cols=[col for col in label_nilai_pegawai.columns if label_nilai_pegawai[col].dtypes=="object"]

object_cols



#Get number of unique entries

object_nunique=list(map(lambda col: label_nilai_pegawai[col].nunique(), object_cols))

d=dict(zip(object_cols, object_nunique))



#print number of unique

sorted(d.items(), key=lambda x:x[1])
#Apply label encoder

label_encoder = LabelEncoder() 



# Encode labels in column 'Satuan Kerja'. 

label_nilai_pegawai['Satuan Kerja']= label_encoder.fit_transform(label_nilai_pegawai['Satuan Kerja'].astype(str)) 

label_nilai_pegawai['Tingkat Pendidikan Akhir']= label_encoder.fit_transform(label_nilai_pegawai['Tingkat Pendidikan Akhir'].astype(str))
label_nilai_pegawai.head()
#Selecting the Prediction Target

y=label_nilai_pegawai.Nilai



#Features

X_Features=['Tingkat Pendidikan Akhir', 'Usia', 'Satuan Kerja']

x=label_nilai_pegawai[X_Features]
#Train Test Split

from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y= train_test_split(x,y, random_state=0)
#Fitting Regression to Training set

from sklearn.linear_model import LinearRegression



regresor=LinearRegression()

regresor.fit(train_x, train_y)
#get predicted 

from sklearn.metrics import mean_absolute_error



val_predictions=regresor.predict(val_x)

print(mean_absolute_error(val_y, val_predictions))