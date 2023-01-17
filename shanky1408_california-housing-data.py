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
demo=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
demo.head()
demo.info()
#missing values
demo.isnull().sum()
demo['ocean_proximity'].value_counts()
from collections import Counter
Counter(demo['ocean_proximity'])
dummy=pd.get_dummies(demo['ocean_proximity'],prefix='ocean').iloc[:,:-1]
data=pd.concat([demo,dummy],axis=1)
data=data.drop('ocean_proximity',axis=1)
data.shape
data.head()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
it=IterativeImputer(estimator=LinearRegression())
new_data=pd.DataFrame(it.fit_transform(data))
new_data.columns=data.columns
new_data.head()
new_data.isnull().sum()
data.describe()
new_data.describe()
#trying imputation with different algorithm (Random Forest) as with the previous algorithm number of bedrooms were imputes as negative
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
it=IterativeImputer(estimator=RandomForestRegressor())
new_data=pd.DataFrame(it.fit_transform(data))
new_data.columns=data.columns
new_data.head()
data.describe()
new_data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(data['total_bedrooms'])
sns.distplot(new_data['total_bedrooms'])
#hypothesis testing
#test of means
#test of variance
#t-test ind
import scipy.stats as stats
stats.ttest_ind(data['total_bedrooms'],new_data['total_bedrooms'],nan_policy='omit')
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()

powerdata=pd.DataFrame(pt.fit_transform(new_data))
powerdata.columns = new_data.columns
powerdata.head()
new_data.hist(figsize=(20,10),bins=10)
plt.show()
powerdata.hist(figsize=(20,10),bins=10)
plt.show()
#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled_data=pd.DataFrame(sc.fit_transform(powerdata))
scaled_data.columns=powerdata.columns
scaled_data.head()
scaled_data.hist(figsize=(20,10),bins=10)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=scaled_data.drop('median_house_value',axis=1)
y=scaled_data['median_house_value']

X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=10)
lr=LinearRegression()
lr.fit(X_train,y_train)
data = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
dummy = pd.get_dummies(data["ocean_proximity"], prefix='Ocean_').iloc[:,:-1]
data = pd.concat([data,dummy], axis=1)
data = data.drop("ocean_proximity", axis=1)
data.shape


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("lr", LinearRegression()),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("poly",PolynomialFeatures(degree=3)),
("lr", LinearRegression()),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
#using XGBooster
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures
X=data.drop('median_house_value',axis=1)
y=data['median_house_value']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)
pipe = Pipeline((
("it", IterativeImputer(estimator = LinearRegression())),
("pt", PowerTransformer()),
("sc",StandardScaler()),
("lr", XGBRegressor(n_estimators=100)),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
from sklearn.model_selection import cross_val_score
scorexgb = cross_val_score(pipe,Xtrain,ytrain,cv=5,scoring='r2')
print(scorexgb)
print('Average R-Square')
print(np.mean(scorexgb))
print('Standard Deviation:',np.std(scorexgb))
import scipy.stats as stats
n=10
xbar = np.mean(scorexgb)
s = np.std(scorexgb)
se = s/np.sqrt(n)
stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)

