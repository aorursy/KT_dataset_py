import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
D1 = pd.read_csv("../input/heart.csv")

D1.head()
D1.dtypes
D1.info()

sns.pairplot(D1)
# Change Datatype Of Columns:- 

D1.sex = D1.sex .astype('object')

D1.fbs = D1.fbs .astype('object')

D1.restecg = D1.restecg .astype('object')

D1.exang = D1.exang .astype('object')

D1.slope = D1.slope .astype('object')

D1.ca = D1.ca .astype('object')

D1.thalthal = D1.thal .astype('object')

D1.target = D1.target .astype('object')

D1.info()
# divide data in Numeric and Cat variable

cat_var = [key for key in dict(D1.dtypes)

             if dict(D1.dtypes)[key] in ['object'] ] # Categorical Varible



numeric_var = [key for key in dict(D1.dtypes)

                   if dict(D1.dtypes)[key]

                       in ['float64','float32','int32','int64']] # Numeric Variable
# check any Extreme value is present in numeric variable

D1.describe()
D1.boxplot(column= numeric_var)
median = D1.loc[D1['trestbps']<140, 'trestbps'].median()

D1.loc[D1.trestbps > 140, 'trestbps'] = np.nan

D1.fillna(median,inplace=True)



median = D1.loc[D1['chol']<246.264026, 'chol'].median()

D1.loc[D1.chol > 246.264026, 'chol'] = np.nan

D1.fillna(median,inplace=True)

import statsmodels.formula.api as smf

import statsmodels.stats.multicomp as multi
# Select imprortant varaible in Catagorical

#ANOVA F Test COVERAGE

model = smf.ols(formula='age ~ target', data=D1)

results = model.fit()

print (results.summary())

# Here The F-statistic is  high

# p-value is to low.
model = smf.ols(formula='cp ~ target', data=D1)

results = model.fit()

print (results.summary())
model = smf.ols(formula='trestbps ~ target', data=D1)

results = model.fit()

print (results.summary())
model = smf.ols(formula='chol ~ target', data=D1)

results = model.fit()

print (results.summary())
model = smf.ols(formula='thalach ~ target', data=D1)

results = model.fit()

print (results.summary())
model = smf.ols(formula='oldpeak ~ target', data=D1)

results = model.fit()

print (results.summary())
model = smf.ols(formula='thal ~ target', data=D1)

results = model.fit()

print (results.summary())
# Start Some Feature Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

D1.head(n = 3)
X = D1.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]  #independent columns

y = D1.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

test = SelectKBest(score_func=chi2, k=2)

y = y.astype('int')
fit = test.fit(X, y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

features = fit.transform(X)

print(features[0:4,:4])
# 1. Linear Reggression

from sklearn.model_selection import train_test_split

X = D1.iloc[:,[0,1,3,4,5,6,7,8,9,10,11,12]]  #independent columns

y = D1.iloc[:,-1] #dependent columns.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# fit a model

from sklearn import linear_model

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predict_train = lm.predict(X_train)

predict_test  = lm.predict(X_test)
# R- Square on Train

lm.score(X_train, y_train)
# R- Square on Test

lm.score(X_test,y_test)
# Mean Square Error

mse = np.mean((predict_train - y_train)**2)

mse
mse = np.mean((predict_test - y_test)**2)

mse