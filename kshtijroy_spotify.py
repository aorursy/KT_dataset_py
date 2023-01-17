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
import matplotlib.pyplot as plt

import seaborn
data= pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

data.head()
data.info()
data.describe()
data.isnull().sum()
data['Artist.Name'].value_counts()[:10].plot(kind='bar')
data[data['Artist.Name']=='Ed Sheeran']
data['Genre'].value_counts().plot(kind='bar')
data[data['Genre']=='dance pop']
plt.figure(figsize=(10,8))

data['Genre'].value_counts().plot(kind='pie',autopct='%1.1f%%')
data['Popularity'].plot(kind='hist')
data['Energy'].plot(kind='hist')
plt.figure(figsize=(8,6))

plt.scatter(x= data['Popularity'],y=data['Energy'])

plt.xlabel('Popularity')

plt.ylabel('Energy')

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(x= data['Popularity'],y=data['Danceability'])

plt.xlabel('Popularity')

plt.ylabel('Danceability')

plt.show()
data1= pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

data1.head()
data1 =data1.drop(['Unnamed: 0','Track.Name','Artist.Name'],axis=1)

x = data1.drop(['Popularity'],axis=1)

y = data1['Popularity']

x= pd.get_dummies(x)
x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size=0.3)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)
from sklearn.metrics import mean_squared_error

print('Mean Squared error :',mean_squared_error(y_test,y_pred))

print('Root mean squared error :', np.sqrt(mean_squared_error(y_test,y_pred)))