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
demo = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
demo.head()
demo.info()
from collections import Counter
Counter(demo.ocean_proximity)
# Encode Categorical Data
data=demo
dummy = pd.get_dummies(data["ocean_proximity"], prefix='Ocean_').iloc[:,:-1]
data = pd.concat([data,dummy], axis=1)
data = data.drop("ocean_proximity", axis=1)
data.shape

data.head()
# # Missing Values
# from sklearn.experimental import enable_iterative_imputer
# # now you can import normally from sklearn.impute
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import LinearRegression
# it = IterativeImputer(estimator=LinearRegression())
# newdata = pd.DataFrame(it.fit_transform(data))
# newdata.columns = data.columns
# newdata.head()

#Missing Values
from sklearn.experimental import enable_iterative_imputer
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
it = IterativeImputer(estimator=RandomForestRegressor())
newdata = pd.DataFrame(it.fit_transform(data))
newdata.columns = data.columns
newdata.head()

newdata.info()
data.describe()
newdata.describe()
from matplotlib import pyplot as plt
import seaborn as sns
sns.distplot(data['total_bedrooms'])
sns.distplot(newdata['total_bedrooms'])
# Hypothesis Testing
# Test of Means
# Test of Variance
import scipy.stats as stats
stats.ttest_ind(data['total_bedrooms'],newdata['total_bedrooms'], nan_policy = 'omit')
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer()
powerdata = pd.DataFrame(pt.fit_transform(newdata))
powerdata.columns = newdata.columns
powerdata.head()

powerdata.skew()
newdata.skew()
powerdata.hist(figsize=(10,10), bins=50)
plt.show()
newdata.hist(figsize=(10,10), bins=50)
plt.show()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaleddata = pd.DataFrame(sc.fit_transform(powerdata))
scaleddata.columns = powerdata.columns
scaleddata.head()

scaleddata.hist(figsize=(10,10), bins=50)
plt.show()
# Incorrect approach
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X = scaleddata.drop('median_house_value', axis = 1)
y = scaleddata['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 10)
lr = LinearRegression()
lr.fit(X_train, y_train)
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
from sklearn.preprocessing import PolynomialFeatures
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
("ploy", PolynomialFeatures(degree = 3)),
("lr", LinearRegression()),
))
pipe.fit(Xtrain,ytrain)
print("Training R2")
print(pipe.score(Xtrain,ytrain))
print("Testing R2")
print(pipe.score(Xtest,ytest))
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

pipe.named_steps['lr'].coef_
from xgboost import XGBRegressor
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

###Cross Validation
from sklearn.model_selection import cross_val_score
scoresxgb = cross_val_score(pipe,Xtrain,ytrain,cv=5,scoring='r2')
print(scoresxgb)
import numpy as np
print('Average R2 of the model')
print(np.mean(scoresxgb))
print('ASD of the model')
print(np.std(scoresxgb))
#Estimate Confidence Interval of R2
import scipy.stats as stats
n=10
xbar = np.mean(scoresxgb)
s = np.std(scoresxgb)
se = s/np.sqrt(n)
stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)
