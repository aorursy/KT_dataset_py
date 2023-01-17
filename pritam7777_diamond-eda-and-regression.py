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
data=pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

print(data.shape)
data.head()
data.isna().sum()
data['price'].value_counts()
print(data['cut'].nunique())

print(data['cut'].value_counts())
data.describe()
data.fillna(0,inplace=True)
data.drop(['Unnamed: 0'],axis=1,inplace=True)

data.head()
data.dtypes
one_hot_encoders=  pd.get_dummies(data)

one_hot_encoders.head()
cols=one_hot_encoders.columns

clean_data=pd.DataFrame(one_hot_encoders,columns=cols)

clean_data.head()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()
numerical=pd.DataFrame(sc_X.fit_transform(clean_data[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=clean_data.index)
numerical.head()
clean_data_standard = clean_data.copy(deep=True)

clean_data_standard[['carat','depth','x','y','z','table']] = numerical[['carat','depth','x','y','z','table']]
clean_data_standard.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

sns.heatmap(clean_data.corr(), annot=True,cmap='RdYlGn')
plt.figure(figsize=(20,20))

sns.heatmap(clean_data_standard.corr(),annot=True,cmap='RdYlGn')
X = clean_data_standard.drop(["price"],axis=1)

y = clean_data_standard.price
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression , Lasso, Ridge
regr=LinearRegression()

regr.fit(X_train,y_train)

y_pred=regr.predict(X_test)

print('accuracy:',regr.score(X_test,y_test))

print('mean_absolute_error:',mean_absolute_error(y_test,y_pred))

print('mean_squared_error:',mean_squared_error(y_test,y_pred))

print('R2_score:',r2_score(y_test,y_pred))
las_reg=Lasso()

las_reg.fit(X_train,y_train)

y_pred=regr.predict(X_test)

print('accuracy:',regr.score(X_test,y_test))

print('mean_absolute_error:',mean_absolute_error(y_test,y_pred))

print('mean_squared_error:',mean_squared_error(y_test,y_pred))

print('R2_score:',r2_score(y_test,y_pred))

rig_reg=Ridge()

rig_reg.fit(X_train,y_train)

y_pred=regr.predict(X_test)

print('accuracy:',regr.score(X_test,y_test))

print('mean_absolute_error:',mean_absolute_error(y_test,y_pred))

print('mean_squared_error:',mean_squared_error(y_test,y_pred))

print('R2_score:',r2_score(y_test,y_pred))