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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.info()
df.describe()
print(df.Fuel_Type.unique())

print(df.Year.unique())
dfnew=df.drop(['Selling_Price'],axis=1)

dfnew

dfnew=dfnew.drop(['Car_Name'],axis=1)

dfnew

dfnew['current_yr']=2020

dfnew

dfnew['no_yrs']=dfnew['current_yr']-dfnew['Year']

dfnew=dfnew.drop(['Year','current_yr'],axis=1)

dfnew

dfnew=pd.get_dummies(dfnew,drop_first=True)

dfnew
sns.set()

plt.plot(df.Selling_Price,df.Kms_Driven,'-')

sns.scatterplot(data=df,x=df.Kms_Driven,y=df.Selling_Price,hue='Fuel_Type')
a=df.corr()

a

b=a.index

z=sns.heatmap(a,annot=True,cmap="RdYlGn")
plt.plot(df.Selling_Price,df.Present_Price)
sns.scatterplot(data=df,x=df.Selling_Price,y=df.Present_Price,hue='Fuel_Type')
y=df.Selling_Price
model=ExtraTreesRegressor()

model.fit(dfnew,y)
model.feature_importances_
check=pd.Series(model.feature_importances_,index=dfnew.columns)

check.nlargest(6).plot(kind='barh')

plt.show()
xtrain,xtest,ytrain,ytest=train_test_split(dfnew,y)

print(xtrain.shape)

print(xtest.shape)

print(ytrain.shape)

print(ytest.shape)

print(df.shape)
model=LinearRegression()
check=model.fit(xtrain,ytrain)

predictions=check.predict(xtest)
sns.distplot(ytest-predictions)
plt.scatter(ytest,predictions)
print('MAE:', metrics.mean_absolute_error(ytest, predictions))

print('MSE:', metrics.mean_squared_error(ytest, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, predictions)))