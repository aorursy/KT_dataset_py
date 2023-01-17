import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.nonparametric.kde import KDEUnivariate

from statsmodels.nonparametric import smoothers_lowess

from pandas import Series, DataFrame

from patsy import dmatrices

from sklearn import datasets, svm
df = pd.read_csv("../input/train.csv",index_col="PassengerId") 
df.head(1)
df.isnull().any()
df=pd.concat([df,pd.get_dummies(df[['Sex','Embarked']], drop_first = True)],axis=1)

df=df.drop(['Cabin','Name','Ticket','Sex','Embarked'],axis=1)
df.head(1)
sur_age=df.Age.dropna().groupby(df.Survived)
sur_age.describe()
fg=plt.figure(figsize=(9,7))

p1=plt.subplot(221)

l1=plt.hist(df.Age.dropna(), 25)

plt.title('Histogram of Age in general')



################################################

p2=plt.subplot(222)

l2=sur_age.hist(bins=30,alpha=0.5)

plt.title('Separated Histogram of Age')

plt.show()
child=df[df.Age<18]
sur_child=child.Age.groupby(df.Survived)
sur_child.describe()
sur_child.median()
adult=df[df.Age>=18]

sur_adult=adult.Age.groupby(df.Survived)

sur_adult.describe()
sur_adult.median()
df.dropna().groupby('Survived').describe()
nonull_df=df.dropna()

nonull_label=nonull_df.Age

nonull_df=nonull_df.drop(['Age','Sex_male','Embarked_Q','Embarked_S'],axis=1)

#nonull_df=nonull_df.drop('Age')

null_df=df[df.isnull().any(axis=1)]

null_df=null_df.drop('Age',axis=1)

null_df.head(1)
nonull_label.head(5)
nonull_df.head(1)
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import normalize

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test=train_test_split (normalize(nonull_df),nonull_label,random_state=1)
linreg=LinearRegression()

linreg.fit(X_train,y_train)

linreg.coef_
#Predictions

y_pred_train=linreg.predict(X_train)

y_pred_train
#error->RMSE

from sklearn import metrics

import numpy as np

np.sqrt(metrics.mean_squared_error(y_train,y_pred_train))
#If apply on test set

y_pred_test=linreg.predict(X_test)

np.sqrt(metrics.mean_squared_error(y_test,y_pred_test))
import seaborn as sns

features_cols=['Survived','SibSp','Parch','Pclass','Fare']

sns.pairplot(pd.concat([nonull_df,nonull_label],axis=1),x_vars=features_cols, y_vars='Age',kind='reg')