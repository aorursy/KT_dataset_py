# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/cmpe343/train.csv')
df_t=pd.read_csv('../input/cmpe343/test.csv')
df.head()
df['Open Date'] = pd.to_datetime(df['Open Date'], format='%m/%d/%Y') 
df_t['Open Date'] = pd.to_datetime(df_t['Open Date'], format='%m/%d/%Y')
df.describe()
df.isnull().sum()
df.drop('Id',axis=1,inplace=True)
df_t.drop('Id',axis=1,inplace=True)
print(df['City Group'].value_counts())
df['City Group'].value_counts().plot(kind='bar')
plt.xlabel('Type of city')
plt.ylabel('No. of restaurants')
plt.title('No. of Restaurants in Big cities and Other cities')
plt.show()
print(df['Type'].value_counts())
df['Type'].value_counts().plot(kind='bar')
plt.xlabel('Type of Restaurants')
plt.ylabel('No. of restaurants')
plt.title('No. of Restaurants of each city')
plt.show()
df1=pd.get_dummies(df,columns=['City'],drop_first=True)
df_t=pd.get_dummies(df_t,columns=['City'],drop_first=True)
x_d= df1.set_index('Open Date', append=False)
x_d=x_d.index.to_julian_date()
x1= df_t.set_index('Open Date', append=False)
x1=x1.index.to_julian_date()
df1['open date']=x_d
df_t['open date']=x1
df1.drop('Open Date',axis=1,inplace=True)
df_t.drop('Open Date',axis=1,inplace=True)
df1['City Group'].replace({'Big Cities':1,'Other':0},inplace=True)
df1['Type'].replace({'IL':0,'FC':1,'DT':2,'MB':3},inplace=True)
df_t['City Group'].replace({'Big Cities':1,'Other':0})
df_t['Type'].replace({'IL':0,'FC':1,'DT':2,'MB':3},inplace=True)
df1

cols
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df_x=ss.fit_transform(df1[['open date']])
df1['open date']=df_x
df1.head()
x=df1.drop('revenue',axis=1)
y=df1['revenue']
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
print('Coefficients are :\n')
list(zip(x.columns,model.coef_))
print('R square value: ',model.score(x,y))