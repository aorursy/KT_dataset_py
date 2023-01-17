#Load the librarys

import pandas as pd #To work with dataset

import numpy as np #Math library

import seaborn as sns #Graph library that use matplot in background

import matplotlib.pyplot as plt #to plot some parameters in seaborn



#Importing the data

df_credit = pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv",index_col=0)

df_credit.head()
#Searching for Missings,type of data and also known the shape of data

print(df_credit.info())
df_credit.isnull().sum()/len(df_credit)
#Looking unique values

print(df_credit.nunique())
sns.countplot(df_credit['Risk'])
sns.countplot(x='Risk', hue='Job', data=df_credit)
sns.countplot(x='Sex', hue='Risk', data=df_credit)
sns.distplot(df_credit['Age'],bins=5,kde=False)
sns.distplot(df_credit[df_credit['Risk']=='good']['Age'], bins=10, kde=False)
sns.distplot(df_credit[df_credit['Risk']=='bad']['Age'], bins=10, kde=False)
sns.pairplot(df_credit,hue='Risk')
df_credit.fillna('Missing', inplace=True)

df_credit.head()
df_credit['Job'] = df_credit['Job'].astype('category')
df_credit_dummy = pd.get_dummies(df_credit, drop_first = True)

df_credit_dummy.head()
X = df_credit_dummy.drop('Risk_good',axis=1)

y = df_credit_dummy['Risk_good']
from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
log = LogisticRegression(solver = 'lbfgs', max_iter = 1000)

log.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth = 50)

dt.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

from sklearn.model_selection import GridSearchCV
y_pred_log = log.predict(X_test)

y_pred_dt = dt.predict(X_test)
print (accuracy_score(y_test,y_pred_log))

print (confusion_matrix(y_test, y_pred_log))

print (classification_report(y_test, y_pred_log))
print (accuracy_score(y_test,y_pred_dt))

print (confusion_matrix(y_test, y_pred_dt))

print (classification_report(y_test, y_pred_dt))
param_grid = { "max_depth" : [3,5,7,9,11,13,15,17,19,21],

              "max_features" : [2,4,6,8,10]}



model = RandomForestClassifier(random_state = 42)



grid_search = GridSearchCV(model, param_grid = param_grid, cv = 5 )
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)

print(grid_search.best_params_)
rf = RandomForestClassifier(max_depth= 7, max_features= 2)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print (accuracy_score(y_test,y_pred_rf))

print (confusion_matrix(y_test, y_pred_rf))

print (classification_report(y_test, y_pred_rf))
y_pred_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()