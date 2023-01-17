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
df=pd.read_csv('/kaggle/input/c001hw1/income_prediction_train.csv')
df
df.columns
df.isnull().sum()
df['income'].mean()
df[df['income']>12712]['age']
import matplotlib.pyplot as plt

plt.scatter(df['age'],df['income'])

plt.show()
plt.scatter(df['height'],df['income'])

plt.show()
dict_1=dict(df.corr()['income'])
list_1=[]

for key,values,in dict_1.items():

    if values<0:

        list_1.append(key)
list_1
df=df.drop(list_1,axis=1)
df
y=df['income']

x=df.drop(['income'],axis=1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,r2_score

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)

score_1=r2_score(y_test,pred_1)
score_1