# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.drop(['Name','Ticket'],axis=1,inplace=True)
train_data.fillna(0,inplace=True)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.drop(['Name','Ticket'],axis=1,inplace=True)
test_data.fillna(0,inplace= True)
%matplotlib inline 
import matplotlib.pyplot as plt
for i in range(0,891):
    if train_data.iloc[i,8] != 0:
        train_data.iloc[i,8] = 1
    
train_data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sex_cat = train_data['Sex']
sex_en = encoder.fit_transform(sex_cat)
from sklearn.preprocessing import OneHotEncoder
en = LabelEncoder()
Em_cat =train_data['Embarked']
Em_en = en.fit_transform(Em_cat)

pd.set_option('display.max_rows' , 700 )
#a.sort_values(['Sex','SibSp','Parch','Age'])



from pandas.plotting import scatter_matrix

#scatter_matrix(train_data[attr],figsize=(20,12))

