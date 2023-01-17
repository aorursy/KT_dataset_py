# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/predict-the-data-scientists-salary-in-india-hackathon"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
df=pd.read_csv('../input/predict-the-data-scientists-salary-in-india-hackathon/Final_Train_Dataset.csv')
df.head()

del df['Unnamed: 0']
df.shape
df.isnull().sum()
del df['job_type']
len(df['job_description'].unique())
len(df['experience'].unique())
df.groupby(['job_description','salary']).size()
del df['job_description']
df.head()
len(df['job_desig'].unique())
del df['job_desig']
df.head()
len(df['key_skills'].unique())
del df['key_skills']
len(df['company_name_encoded'].unique())
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['experience']=le.fit_transform(df['experience'])
df['location']=le.fit_transform(df['location'])
df['salary']=le.fit_transform(df['salary'])
from sklearn.model_selection import train_test_split
x=df.drop('salary',axis=1)
y=df['salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=30)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pred)
