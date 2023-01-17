import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/TRAIN.csv')
test  = pd.read_csv('../input/TEST.csv')
train_orginal = train.copy()
test_orginal  = test.copy()
train.shape
train.columns
train.dtypes
train.shape

test.shape
test.columns
test.dtypes
train.head()
train["Network type subscription in Month 1"].value_counts()
train["Network type subscription in Month 2"].value_counts()
train["Most Loved Competitor network in in Month 1"].value_counts()
train["Most Loved Competitor network in in Month 2"].value_counts()
train["Churn Status"].value_counts()
Tech=pd.crosstab(train['Network type subscription in Month 1'],train['Churn Status'])
Tech.div(Tech.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
Tech=pd.crosstab(train['Network type subscription in Month 2'],train['Churn Status'])
Tech.div(Tech.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.figure(1)
plt.subplot(131)
train['Most Loved Competitor network in in Month 1'].value_counts().plot.bar(figsize=(24,6), title= 'Competitor_at_stage1')

plt.subplot(132)
train['Most Loved Competitor network in in Month 2'].value_counts().plot.bar(title= 'Competitor_at_stage2')

plt.show()
train["Churn Status"].value_counts().plot.bar(title= 'Churn rate')
train["Total Call centre complaint calls"].value_counts().plot.bar(title= 'Churn rate')

# Print correlation matrix
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train=train.drop('Customer tenure in month',axis=1)
train.head()
train=train.drop('Total Spend in Months 1 and 2 of 2017',axis=1)
train.head()
train.shape
train=train.dropna(how='all')
train.isnull().sum()
train['Network type subscription in Month 1'].fillna(train['Network type subscription in Month 1'].mode()[0], inplace=True)
train['Network type subscription in Month 2'].fillna(train['Network type subscription in Month 2'].mode()[0], inplace=True)
train['Most Loved Competitor network in in Month 1'].fillna(train['Most Loved Competitor network in in Month 1'].mode()[0], inplace=True)
train['Most Loved Competitor network in in Month 2'].fillna(train['Most Loved Competitor network in in Month 2'].mode()[0], inplace=True)

train.isnull().sum()
train=train.drop('Customer ID',axis=1)

train.shape
test.isnull().sum()
test['Network type subscription in Month 1'].fillna(test['Network type subscription in Month 1'].mode()[0], inplace=True)
test['Network type subscription in Month 2'].fillna(test['Network type subscription in Month 2'].mode()[0], inplace=True)
X = train.drop('Churn Status',1)
y = train["Churn Status"]

X=pd.get_dummies(X)
# Importing packages for cross validation and logistic regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y)
y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm
y_pred
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 10)
classifier.fit(X, y)

y_pred = classifier.predict(X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
y_pred = classifier.predict(X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm
y_pred
submission = test['Customer ID']
test = test.drop('Customer ID',1)
# Adding dummies to the dataset
test_pred=pd.get_dummies(test)
test_pred=test_pred.drop('Total Spend in Months 1 and 2 of 2017',axis=1)
test_pred=test_pred.drop('Customer tenure in month',axis=1)
chur_pred = classifier.predict(test_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm
submission_result=pd.read_csv("../input/sample submission.csv")

submission_result['Customer ID']=submission
submission_result['Churn Status']=chur_pred


submission_result['Churn Status'] = submission_result['Churn Status'].astype(int)

submission_result.to_csv("../input/sample submission.csv",index=False)
