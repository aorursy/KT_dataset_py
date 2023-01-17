# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.shape
data.dtypes
data['salary']=data['salary'].fillna(data['salary'].median())
data=data.drop('sl_no',axis=1)
data.head()
c=data.corr()

c
x=data.drop('salary',axis=1)
x
y=data['salary'].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=10)
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)
lr.predict([[1,84.00,0.90,2,64.50,2,0,86.04,0,59.42,1]])
ytest[0]


from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

data['status']= label_encoder.fit_transform(data['status']) 

  
data.head()