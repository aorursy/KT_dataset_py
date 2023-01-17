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
data=pd.read_csv('/kaggle/input/kc-housesales-data/kc_house_data.csv',parse_dates=['date'])

data.head()
print(data.shape);

data.isnull().any()
data.describe()
data.shape
data.info()
data.drop('date',axis=1,inplace=True);

data.drop('id',axis=1,inplace=True);
data['price']=data['price'].map(lambda i: np.log(i) if i > 0 else 0);

data['sqft_lot']=data['sqft_lot'].map(lambda i: np.log(i) if i > 0 else 0);

data['view']=data['view'].map(lambda i: np.log(i) if i > 0 else 0);

data['yr_renovated']=data['yr_renovated'].map(lambda i: np.log(i) if i > 0 else 0);

data['sqft_lot15']=data['sqft_lot15'].map(lambda i: np.log(i) if i > 0 else 0);

data.skew()
target_col=['price'];

features=list(set(list(data.columns))-set(data[target_col]))

#Normalizing the features between 0 and 1

data[features]=data[features]/data[features].max()

data.describe()
data.skew()
X=data[features].values;

y=data[target_col].values;
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

fig=plt.figure(figsize=(20,6));

plt.scatter(data['sqft_living'],y);

plt.xlabel('sqft_living');

plt.ylabel('price');

plt.title('LogPrice vs SqftLiving')
corr=data.corr();

fig=plt.figure(figsize=(20,6));

sns.heatmap(corr)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train,y_valid= train_test_split(X,y,test_size=0.2,random_state=0);



model=LinearRegression(fit_intercept=True);

model.fit(X_train,y_train);

pred_valid=model.predict(X_valid);
print(model.coef_)

print(model.intercept_)
pred_train=model.predict(X_train)

print(r2_score(y_train,pred_train));

print(r2_score(y_valid,pred_valid));
from sklearn.ensemble import RandomForestRegressor

model1=RandomForestRegressor(n_estimators=100,random_state=0);

model1.fit(X_train,y_train);

pred_rf_train=model1.predict(X_train);

pred_rf=model1.predict(X_valid);

print('Train_score',r2_score(y_train,pred_rf_train));

print('Test_score', r2_score(y_valid,pred_rf));