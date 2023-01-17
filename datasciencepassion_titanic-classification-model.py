# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

def missingdata(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    ms= ms[ms["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(ms.index, ms["Percent"],color="blue",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return ms

missingdata(train_df)
missingdata(test_df)
test_df.shape
test_df['Age'].mean()
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)
train_df.drop(['Cabin'],axis=1,inplace=True)

test_df.drop(['Cabin'],axis=1,inplace=True)



# Filling missing values of Age with median in train and test dataset.

test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
print('check the missing value in train data')

print(train_df.isnull().sum())

print('___'*10)

print('check the missing value in test data')

print(test_df.isnull().sum())

data_combined=[train_df,test_df]
# Creating new column Family size



for dataset in data_combined:

    dataset['Familysize']=dataset['SibSp']+dataset['Parch']+1
# To seperate title from passenger names.

import re

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in data_combined:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in data_combined:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Creating bins for Age features.



for dataset in data_combined:

    dataset['Age_bin']=pd.cut(dataset['Age'],bins=[0,12,20,40,120],labels=['Children','Teenage','Adult','Elder'])
# Creating bins for Fare features.



for dataset in data_combined:

    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',

                                                                                      'Average_fare','high_fare'])
for dataset in data_combined:

    drop_column = ['Age','Fare','Name','Ticket']

    dataset.drop(drop_column, axis=1, inplace = True)
train_df.drop(['PassengerId'],axis=1,inplace=True)
test_df.head()
# Creating dummies for variables.



train_df = pd.get_dummies(train_df, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])

test_df = pd.get_dummies(test_df, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])
train_df.head()
# Correlation 



sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
# Building Model

from sklearn.model_selection import train_test_split #for split the data

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix

all_features = train_df.drop("Survived",axis=1)

Targeted_feature = train_df["Survived"]

X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature,test_size=0.3,random_state=42)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
# Logistic Regression.



from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

prediction_lr=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))

kfold=KFold(n_splits=10,random_state=22)

result_lr=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score for Logistic REgression is:',round(result_lr.mean()*100,2))

y_pred=cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix', y=1.05, size=15)





# Random Forests



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion='gini', n_estimators=700,

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto',oob_score=True,

                             random_state=1,n_jobs=-1)

model.fit(X_train,y_train)

prediction_rm=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of the Random Forest Classifier is',round(accuracy_score(prediction_rm,y_test)*100,2))

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_rm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix', y=1.05, size=15)
# Support Vector Machines



from sklearn.svm import SVC, LinearSVC



model = SVC()

model.fit(X_train,y_train)

prediction_svm=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of the Support Vector Machines Classifier is',round(accuracy_score(prediction_svm,y_test)*100,2))

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_svm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score for Support Vector Machines Classifier is:',round(result_svm.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix', y=1.05, size=15)
# AdaBoost Classifier



from sklearn.ensemble import AdaBoostClassifier

model= AdaBoostClassifier()

model.fit(X_train,y_train)

prediction_adb=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of the AdaBoostClassifier is',round(accuracy_score(prediction_adb,y_test)*100,2))

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_adb=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score for AdaBoostClassifier is:',round(result_adb.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix', y=1.05, size=15)
# Gradient Boosting Classifier.



from sklearn.ensemble import GradientBoostingClassifier

model= GradientBoostingClassifier()

model.fit(X_train,y_train)

prediction_gbc=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of the Gradient Boosting Classifier is',round(accuracy_score(prediction_gbc,y_test)*100,2))

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_gbc=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')

print('The cross validated score for Gradient Boosting Classifier is:',round(result_gbc.mean()*100,2))

y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)

sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix', y=1.05, size=15)
# Model Evaluation 



models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 

              'Random Forest', 'AdaBoostClassifier', 

              'Gradient Boosting', 

              ],

    'Score': [result_svm.mean(), result_lr.mean(), 

              result_rm.mean(),  result_adb.mean(), 

              result_gbc.mean()]})

models.sort_values(by='Score',ascending=False)
# Hyperparameter Tuning



train_X = train_df.drop("Survived", axis=1)

train_Y=train_df["Survived"]

test_X  = test_df.drop("PassengerId", axis=1).copy()

train_X.shape, train_Y.shape, test_X.shape
# Random Forest Classifier Parameters tunning 

from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()

n_estim=range(100,1000,100)



## Search grid for optimal parameters

param_grid = {"n_estimators" :n_estim}





model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_rf.fit(train_X,train_Y)







# Best score

print(model_rf.best_score_)



#best estimator

model_rf.best_estimator_
# Applying the estimator which we get from tuning of Random Forest.



from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=400,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)



random_forest.fit(train_X,train_Y)

Y_pred_rf=random_forest.predict(test_X)

acc_random_forest = round(random_forest.score(train_X, train_Y) * 100, 2)



print("Important features")

pd.Series(random_forest.feature_importances_,train_X.columns).sort_values(ascending=True).plot.barh(width=0.8)

print('__'*5)

print(acc_random_forest)
# Final Submission.



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_rf})
submission.head(10)