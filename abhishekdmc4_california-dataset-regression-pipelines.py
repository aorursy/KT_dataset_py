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
demo=pd.read_csv("../input/california-housing-prices/housing.csv")

demo.head()
#Values_counts

from collections import Counter

Counter(demo.ocean_proximity)

# Nominal data because no order in this data --- have to make dummies encoding

# ordinal data just normal encoding like high:3, medium:2, low:1
#Encode categorical data

data=demo

dummy = pd.get_dummies(data["ocean_proximity"], prefix='Ocean_').iloc[:,:-1]

data = pd.concat([data,dummy], axis=1)

data = data.drop("ocean_proximity", axis=1) 

data.shape



# read about feature hasher for encoding (used when there are a lot of classes in a categorical col)

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html

data.head()
data.info()
#Missing values

# using iterative imputer to impute values



from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

it = IterativeImputer(estimator=LinearRegression())

newdata = pd.DataFrame(it.fit_transform(data))

newdata.columns = data.columns

newdata.head()

newdata.info()
# check if imputation was correct (we plot distribution)

data.describe()
newdata.describe()
# total_bedroom minimum has gone down in negative so we look for something other than linear regression

# lets try Random forests



from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

it = IterativeImputer(estimator=RandomForestRegressor())

newdata = pd.DataFrame(it.fit_transform(data))

newdata.columns = data.columns

newdata.head()
newdata.describe()
# we can see histograms of data and see there are 2 visible clusters,

# Semi supervised learning (supervised+unsupervised)

# so seperate the data using clustering and treat clusters differently (not in this notebook)
# imputation validation

from matplotlib import pyplot  as plt 

import seaborn as sns

sns.distplot(data['total_bedrooms'])
from matplotlib import pyplot  as plt 

import seaborn as sns

sns.distplot(newdata['total_bedrooms'])
# hypothesis testing 

# test of variance

# t test

import scipy.stats as stats

stats.ttest_ind(data['total_bedrooms'],newdata['total_bedrooms'],nan_policy='omit')



# very high P value 

# H0: mean b4 imputation = mean after imputation

# H1: not

# if P<5% reject null

# P is very high so we fail to reject null hypothesis

# transformation - to deal with skewness (Box-cox transform), also deal with Outliers

#make data more gaussian

from sklearn.preprocessing import PowerTransformer #normalises the data

pt=PowerTransformer()

powerdata=pd.DataFrame(pt.fit_transform(newdata))

powerdata.columns=newdata.columns

powerdata.head()

newdata.hist(figsize=(10,10),bins=50)

plt.show()
powerdata.hist(figsize=(10,10),bins=50)

plt.show()
scaleddata.hist(figsize=(10,10),bins=50)

plt.show()
newdata.skew()
powerdata.skew()
#Test train split before scaling and transforming (if not data leak happens)



#Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scaleddata = pd.DataFrame(sc.fit_transform(powerdata))

scaleddata.columns = powerdata.columns

scaleddata.head()

# doing everything again with Pipeline

data = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")

dummy = pd.get_dummies(data["ocean_proximity"], prefix='Ocean_').iloc[:,:-1]

# one hot encoder if want to do this in pipeline

data = pd.concat([data,dummy], axis=1)

data = data.drop("ocean_proximity", axis=1)

data.shape

#Iteration 1 LR

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

("it", IterativeImputer(estimator = LinearRegression())),# use random forest for better result

("pt", PowerTransformer()),

("sc",StandardScaler()),

("lr", LinearRegression()),

))

pipe.fit(Xtrain,ytrain)

print("Training R2")

print(pipe.score(Xtrain,ytrain))

print("Testing R2")

print(pipe.score(Xtest,ytest))

# extract steps from pipeline

pipe.named_steps['lr'].coef_
#Iteration 2 LR(poly)

from sklearn.preprocessing import PolynomialFeatures

X=data.drop('median_house_value',axis=1)

y=data['median_house_value']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=10)

pipe = Pipeline((

("it", IterativeImputer(estimator = LinearRegression())),

("pt", PowerTransformer()),

("sc",StandardScaler()),

("poly",PolynomialFeatures(degree=3)),# X^3 +X^2 highest power of equation is 3 # feature engg technique # making new features like x^2, x^3 ,x^0 etc

("lr", LinearRegression()),

))

pipe.fit(Xtrain,ytrain)

print("Training R2")

print(pipe.score(Xtrain,ytrain))

print("Testing R2")

print(pipe.score(Xtest,ytest))

#Iteration 3 #Booster

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

print("Average R2 of model")

print(np.mean(scoresxgb))

print("SD of model")

print(np.std(scoresxgb))

#read about repeted K fold cross validation

# https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/

#Estimate Confidence Interval of R2

import scipy.stats as stats

n=10 # sample size

xbar = np.mean(scoresxgb)

s = np.std(scoresxgb)

se = s/np.sqrt(n) #se std error

stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)


