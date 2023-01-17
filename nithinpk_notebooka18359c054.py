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
df=pd.read_csv('../input/california-housing-prices/housing.csv')
df.head()
100*round(df.isnull().sum()/len(df.index),2)
from collections import Counter
Counter(df.ocean_proximity)
df.ocean_proximity.value_counts()

dummy = pd.get_dummies(df[['ocean_proximity']], drop_first=True)
df = pd.concat([df, dummy], axis=1)

df=df.drop('ocean_proximity',axis=1)
# Setting display option
pd.set_option('display.max_columns', 500)  
pd.set_option('display.max_rows', 500)  
pd.set_option('display.width', 1000) 
df.head()
df.to_csv('afterdummy.csv')
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=df.drop('median_house_value',axis=1)
y=df['median_house_value']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=10)
pipe=Pipeline((
("it", IterativeImputer(estimator=LinearRegression())),
("pt",PowerTransformer()),
("sc",StandardScaler()),
("lr", LinearRegression()),

))
pipe.fit(Xtrain,ytrain)
print("train R2 Score")
print(pipe.score(Xtrain,ytrain))
print("test R2 Score")
print(pipe.score(Xtest,ytest))
from sklearn.preprocessing import PolynomialFeatures
X=df.drop('median_house_value',axis=1)
y=df['median_house_value']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=10)
pipe=Pipeline((
("it", IterativeImputer(estimator=LinearRegression())),
("pt",PowerTransformer()),
("sc",StandardScaler()),
("poly",PolynomialFeatures(degree=2)),
("lr", LinearRegression()),

))
pipe.fit(Xtrain,ytrain)
print("train R2 Score")
print(pipe.score(Xtrain,ytrain))
print("test R2 Score")
print(pipe.score(Xtest,ytest))
# This piece of code is used to extract steps from pipe
pipe.named_steps['lr'].coef_
from xgboost import XGBRegressor
X=df.drop('median_house_value',axis=1)
y=df['median_house_value']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=10)
pipe=Pipeline((
("it", IterativeImputer(estimator=LinearRegression())),
("pt",PowerTransformer()),
("sc",StandardScaler()),
("lr", XGBRegressor()),

))
pipe.fit(Xtrain,ytrain)
print("train R2 Score")
print(pipe.score(Xtrain,ytrain))
print("test R2 Score")
print(pipe.score(Xtest,ytest))
from sklearn.model_selection import cross_val_score
scoresxgb=cross_val_score(pipe,Xtrain,ytrain,cv=5,scoring='r2')
print(scoresxgb)
# average r2 of the model
print("average r2 of the model")
print(np.mean(scoresxgb))
print("sd of the model")
print(np.std(scoresxgb))
# Missing values 
from sklearn.experimental import enable_iterative_imputer
df.head()
df.shape
df.columns
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
it=IterativeImputer(estimator=LinearRegression())
df1=pd.DataFrame(it.fit_transform(df))

100*round(df1.isnull().sum()/len(df1.index),2)
df.head()
cl=df.columns

cl
df1.columns=cl
df1.head()
df1.info()
df.describe()
df1.describe()
from sklearn.ensemble import RandomForestRegressor
it=IterativeImputer(estimator=RandomForestRegressor())
df2=pd.DataFrame(it.fit_transform(df))
df2.head()
df2.columns=cl
df2.head()
df2.describe()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(df2['total_bedrooms'])
sns.distplot(df['total_bedrooms'])
import scipy.stats as stats
stats.ttest_ind(df['total_bedrooms'],df2['total_bedrooms'],nan_policy='omit')
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()
df3_powerdata=pd.DataFrame(pt.fit_transform(df2))
df3_powerdata.columns=cl
df3_powerdata.head()
df2.hist(figsize=(10,10),bins=50)
plt.show()
df3_powerdata.hist(figsize=(10,10),bins=50)
plt.show()
df2.skew()
df3_powerdata.skew()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df4_scaled=pd.DataFrame(sc.fit_transform(df3_powerdata))
df4_scaled.head()
df4_scaled.columns=cl
df4_scaled.head()
df4_scaled.hist(figsize=(10,10),bins=50)
plt.show()
# Trail test spli
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=df4_scaled.drop('median_house_value',axis=1)
y=df4_scaled['median_house_value']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=10)
lr1=LinearRegression()
lr1.fit(Xtrain,ytrain)
lr1.score(Xtrain,ytrain)