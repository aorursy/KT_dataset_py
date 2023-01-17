#Import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

%load_ext autoreload
%autoreload 2

#Load DATA - Train and Test
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#EDA
print('The shape of train data:',df_train.shape)
print('The shape of test data:',df_test.shape)
df_train.columns
df_test.columns
df_train.describe
df_train.info()
df_test.info()
df_train.head(30)
by_class = df_train.groupby(['Pclass','Embarked'])
count_Survived = by_class['Survived'].count()
print(count_Survived)
by_age = df_train.groupby(['Age'])
ccount_survived_age = by_age['Survived'].count()
print(ccount_survived_age.head(2))
#EDA - Graphs
df_train.Survived.value_counts().plot(kind='bar',alpha=0.9)
plt.title('survived = 1 ')
plt.scatter(range(df_train.shape[0]),np.sort(df_train.Age),alpha=0.4)
plt.title('age distribution')
df_train.Pclass.value_counts().plot(kind='barh',alpha=0.5)
plt.title('Pclass')
df_train.Embarked.value_counts().plot(kind='bar',alpha=0.5)
plt.title('Embarked')
plt.scatter(range(df_train.shape[0]),np.sort(df_train.Fare),alpha=0.6)
plt.title('Fare distribution')

train_male = df_train.Survived[df_train.Sex == 'male'].value_counts()

train_female = df_train.Survived[df_train.Sex == 'female'].value_counts()

ind = np.arange(2)
width = 0.3
fig, ax = plt.subplots()
male = ax.bar(ind, np.array(train_male), width, color='r')
female = ax.bar(ind+width, np.array(train_female), width, color='b')
ax.set_ylabel('Count')
ax.set_title('DV count by Gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('DV=0', 'DV=1'))
ax.legend((male[0], female[0]), ('Male', 'Female'))
plt.show()

df_train.columns

from sklearn.preprocessing import Imputer

train_cats(df_train)
df_train.Sex.cat.categories
df_train.Sex = df_train.Sex.cat.codes
df_train.Embarked.cat.categories
df_train.Embarked = df_train.Embarked.cat.codes
by_sex_class = df_train.groupby(['Sex','Pclass'])
def impute_median(series):
    return series.fillna(series.mean())
df_train.Age = by_sex_class.Age.transform(impute_median)
df_train.Age.tail()
X_train = df_train[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
y_train = df_train['Survived']
X_test = df_test[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
X_trn,X_val,y_trn,y_val = train_test_split(X_train,y_train,test_size=0.469,random_state=42)
m = lm.LogisticRegression()
m.fit(X_trn,y_trn)
pred = m.predict(X_val)
pred
#my_submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': pred})
#my_submission.to_csv('submission_rakesh.csv', index=False)

submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": pred
    })
submission

print(submission.to_string())
from sklearn.metrics import accuracy_score

accuracy_score(y_val,pred)
X_train = df_train[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
y_train = df_train['Survived']
X_test = df_test[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_trn,y_trn)
Y_pred = random_forest.predict(X_val)
Y_pred
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
print(submission.to_string())
accuracy_score(y_val,Y_pred)
