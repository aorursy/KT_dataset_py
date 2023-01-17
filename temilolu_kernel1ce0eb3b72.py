#For working with dataframes and numbers

import pandas as pd
import numpy as np


#For Exploratory data analysis
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
%matplotlib inline

#Modelling and eveluation
import sklearn.model_selection as ms
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sklm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


#import datasets
train = pd.read_csv("../input/train_technidus_clf.csv")
test = pd.read_csv("../input/test_technidus_clf.csv")
#Conduct mini data check/analysis

test.shape
train.head()
test.shape
#Check data for missing values
#train.isnull().sum()
train.isnull().sum()
#Check category split
#train.BikeBuyer.value_counts()
train.BikeBuyer.value_counts()
#Univariate continuous- Histogram
train["AveMonthSpend"].plot.hist(color='blue',bins=50)
plt.show()
#Univariate categorical- Countplot(Barplot)
f, ax = plt.subplots(figsize=(8, 4))
sns.countplot("Occupation", data=train)
#Bivariate analysis 
#Continuous and categorical (Box plot)
sns.boxplot('BikeBuyer','AveMonthSpend', data=train)
#Bivariate analysis 
#Continuous and continuous
plt.scatter('AveMonthSpend',"YearlyIncome", data = train)
#Example is generating Age column from Birth date variable
train.BirthDate.head(2)
train['Birthdate_int'] = train.BirthDate.str[-4:]
train  = train.dropna(subset=['Birthdate_int'])
train['Birthdate_int'] = train['Birthdate_int'].astype(int)
train['Birthdate_int'].head()
train['today_date'] = 1998
train['Age'] = train['today_date']-train['Birthdate_int']
train['Age'].head()
#Deletion
#For example delete age greater than 90 years
train['Age'] = train['Age']<90
#Example of Binning
bins = [0,25,45,55,100]
labels = ["0-24","25-44", "45-55","56-100"]
train["Agebin"] = pd.cut(train.Age,bins = bins,labels=labels)
#train.Agebin.value_counts()
train[['Age','Agebin']].head()
#Deletion
#Pairwise
#train = train.dropna(axis=1,how='any')
#listwise
#train = train.dropna(axis=1,how='all')

#Mean/mode imputation
train['Age'].fillna(train['Age'].mean(), inplace=True) 
train['Age'].fillna(train['Age'].mode()[0], inplace=True) 
corr= train.corr()
#corr
f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(corr,cmap='coolwarm',linewidths=2.0, annot=True)
#Convert selected variables to array which scikit learn recognises
X = train[['NumberCarsOwned',
          'NumberChildrenAtHome',
          'YearlyIncome','CountryRegionName']]
y = train['BikeBuyer']
Xb=test[['NumberCarsOwned',
          'NumberChildrenAtHome',
          'YearlyIncome','CountryRegionName']]
#Convert categorical variables to numerical through one-hotencoding
X=pd.get_dummies(X)
Xb=pd.get_dummies(Xb)
#Holdout 30%
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
#Build various models
#Logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
#model = LogisticRegression()
model.fit(x_train, y_train)

#Evaluate logistic regression
pred_cv = model.predict(x_cv)
score = accuracy_score(y_cv,pred_cv)
print('accuracy_score',score)
#Build Random forest
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(random_state=1, max_depth=110,n_estimators= 1400,class_weight="balanced")
model1.fit(X,y)
#Evaluate RF
pred_cv = model.predict(x_cv)
score = accuracy_score(y_cv,pred_cv)
print('accuracy_score',score)
#Save for submission
test['BikeBuyer']=model1.predict(Xb)
test['CustomerID']= test['CustomerID']
test['BikeBuyer'] = test['BikeBuyer'].astype(int)
test[['CustomerID','BikeBuyer']].head(10)
test[['CustomerID','BikeBuyer']].to_csv('perfrect_score2.csv',index= False)

