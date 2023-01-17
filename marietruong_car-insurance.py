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
# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data prepro
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
#modelling

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
sample = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
train
plt.figure()
plt.title('Responses')
sns.barplot(x = train.groupby('Response').count().index, y = train.groupby('Response').count().id)

plt.figure()
plt.title('Gender')
sns.barplot(x = train.groupby('Gender').mean().index, y = train.groupby('Gender').mean().Response)
plt.figure()
plt.title('Age of customers ')
sns.distplot(train.Age[train.Response == 0], color = 'red', hist = False, label = 'No')
sns.distplot(train.Age[train.Response == 1], color = 'green', hist = False, label = 'Yes')
sns.distplot(train.Age, color = 'blue', hist = False, label = 'All')
plt.figure(figsize = (20,10))
plt.title('Driving license')
plt.subplot(1,2,1)
sns.barplot(x = train.groupby('Driving_License').count().index, y = train.groupby('Driving_License').count().id)
plt.subplot(1,2,2)
sns.barplot(x = test.groupby('Driving_License').count().index, y = test.groupby('Driving_License').count().id)
plt.figure(figsize = (20,10))
plt.title('Link between age and annual Premium')
sns.barplot(x = train.Age, y = train.Annual_Premium)
plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.barplot(x = train.groupby('Region_Code').count().index, y = train.groupby('Region_Code').count().id)
plt.ylabel('Number')
plt.subplot(1,2,2)
sns.barplot(x = train.groupby('Region_Code').mean().index, y = train.groupby('Region_Code').mean().Response)

plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
plt.title('Proportion of customers already insured')
plt.ylabel('Count')
sns.barplot(x = train.groupby('Previously_Insured').count().index, y = train.groupby('Previously_Insured').count().id)
plt.subplot(1,2,2)
plt.title('Response of people who are already insured')
sns.barplot(x = train.groupby('Previously_Insured').mean().index, y = train.groupby('Previously_Insured').mean().Response)

plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
plt.title('Vehicle age distribution')
sns.barplot(x = train.groupby('Vehicle_Age').count().index, y = train.groupby('Vehicle_Age').count().id)
plt.ylabel('Count')
plt.subplot(1,2,2)
plt.title('Response of customers according to vehicle age ')
sns.barplot(x = train.groupby('Vehicle_Age').mean().index, y = train.groupby('Vehicle_Age').mean().Response)


plt.title('Annual premium according to vehicle age')
sns.barplot(x = train.groupby('Vehicle_Age').mean().index, y = train.groupby('Vehicle_Age').mean().Annual_Premium)

plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
plt.title('Proportion of customers whose vehicle have been damaged before')
sns.barplot(x = train.groupby('Vehicle_Damage').count().index, y = train.groupby('Vehicle_Damage').count().id)
plt.subplot(1,2,2)
plt.title('Response of customers according to vehicle damage')
sns.barplot(x = train.groupby('Vehicle_Damage').mean().index, y = train.groupby('Vehicle_Damage').mean().Response)

plt.figure(figsize =(20,10))
plt.title('Annual premium for customers who say yes vs for those who say no')
sns.distplot(train[train.Response == 0].Annual_Premium, hist = False, label = 'No', color = 'red')
sns.distplot(train[train.Response == 1].Annual_Premium, hist = False, label = 'Yes', color = 'green')
plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
sns.barplot(x = train.groupby('Policy_Sales_Channel').count().index, y = train.groupby('Policy_Sales_Channel').count().id)
plt.subplot(1,2,2)
sns.barplot(x = train.groupby('Policy_Sales_Channel').mean().index, y = train.groupby('Policy_Sales_Channel').mean().Response)


plt.figure()
plt.title('Vintage')
sns.distplot(train.Vintage, hist = False, label = 'All')
sns.distplot(train[train.Response == 1].Vintage, hist = False, label = 'Yes', color = 'green')
sns.distplot(train[train.Response == 0].Vintage, hist = False, label = 'No', color = 'red')


sns.heatmap(train.corr())
cat = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Previously_Insured']
num = ['Age', 'Annual_Premium']
X = train[cat + num]
y = train['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_test
test2 = test[cat+num]
test2
#Minimum prepro
encodage = make_column_transformer((OneHotEncoder(), cat))
prepro = make_pipeline(encodage)
# Prepro Encodage + Standard
encodage1 = make_column_transformer((OneHotEncoder(), cat), (StandardScaler(), num))
prepro1 = make_pipeline(encodage1)
def evaluation(prepro, X):
    X_pre = prepro.fit_transform(X)
    return cross_val_score(DecisionTreeClassifier(), X_pre, y_train, scoring = 'roc_auc').mean(axis = 0)
evaluation(prepro, X_train)
evaluation(prepro1, X_train)
cat1 = ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Previously_Insured', 'Region_Code', 'Policy_Sales_Channel']
num1 = ['Age', 'Annual_Premium']
X2 = train[cat1+num1]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y)
encodage2 = make_column_transformer((OneHotEncoder(), cat1))
prepro2 = make_pipeline(encodage2)
evaluation(prepro2, X_train2)
X_pre = prepro.fit_transform(X_train)
to_pred = prepro.transform(test2)
to_pred
def evaluation2(model):
    train_sizes, train_scores, val_scores = learning_curve(model, X_pre, y_train, train_sizes = np.linspace(0.1,1,10), scoring = 'roc_auc' )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'train')
    plt.plot(train_sizes, val_scores.mean(axis = 1), label = 'val')
    print (val_scores.mean(axis=1)[-1])  
    plt.title(model)
    plt.legend()
    plt.figure()
model1 = LogisticRegression()
grid = GridSearchCV(model1, param_grid = {'C': np.logspace(-4, 2, 10)}, scoring = 'roc_auc')
grid.fit(X_pre, y_train)
print((grid.best_score_, grid.best_params_))
model2 = RandomForestClassifier()
grid = GridSearchCV(model2, param_grid = {'n_estimators' : [10, 50, 100, 150]}, scoring = 'roc_auc')
grid.fit(X_pre, y_train)
print((grid.best_score_, grid.best_params_))
model2.fit(X_pre, y_train)

submission = pd.Series(data = model2.predict_proba(to_pred)[:,1], index = test['id'], name = 'Response')
submission.to_csv('/kaggle/working/insurance.csv')