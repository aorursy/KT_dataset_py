# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df=pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.head()
df.info()
df.isnull().sum()
df.shape
round(100*(df.isnull().sum()/len(df.index)),2)
UUU=df[['Country','Year','Status']]

df.drop(['Country','Year'],axis=1,inplace=True)
df.drop(['Status'],axis=1,inplace=True)
from sklearn.impute import KNNImputer

imputer=KNNImputer(n_neighbors=7)

YY=imputer.fit_transform(df)
df.columns
imputer=pd.DataFrame(YY,columns=['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol',

       'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',

       'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',

       ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',

       ' thinness 5-9 years', 'Income composition of resources', 'Schooling'])
imputer.head()
imputer.isnull().sum()
imputer.shape
result=pd.concat([UUU,imputer],axis=1)
result.head()
result['Country'].nunique()
result['Status'].unique()
Country_dummy=pd.get_dummies(result['Country'])
status_dummy=pd.get_dummies(result['Status'])
result.drop(['Country','Status'],inplace=True,axis=1)
result=pd.concat([result,Country_dummy,status_dummy],axis=1)
result.info()
result.head()
Life=result['Life expectancy ']
Life.head()
result.drop('Life expectancy ',axis=1,inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(result, Life, test_size=0.30, random_state=101)
from sklearn.linear_model import LinearRegression
Linear_model= LinearRegression()
Linear_model.fit(X_train,y_train)
predictions1=Linear_model.predict(X_test)
predictions1[:10]
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,predictions1)**(0.5))
from sklearn.metrics import r2_score
r2_score(y_test,predictions1)
from sklearn.linear_model import Ridge
ridge_model=Ridge()
ridge_model.fit(X_train,y_train)
predictions2=ridge_model.predict(X_test)
print(mean_squared_error(y_test,predictions2)**(0.5))
from sklearn.linear_model import Lasso
lasso_model=Lasso(alpha=0.00000001)

# Alpha value here was selected after choosing 8 different combinations like 0.1,0.001,0.0001...etc.
lasso_model.fit(X_train,y_train)
predictions3=lasso_model.predict(X_test)
print(mean_squared_error(y_test,predictions3)**(0.5))