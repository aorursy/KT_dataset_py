# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv')
training_data.head()
training_data.isnull().values.any()
#gives birdeye view of columns which might have null values

sns.heatmap(training_data.isnull(),yticklabels=False,cbar=False)
sns.distplot(training_data['Age'].dropna(),kde=False,bins=30)
sns.barplot(x='Pclass',y='Fare',data=training_data,ci=None)
sns.boxplot(x='Pclass',y='Age',data=training_data)
def compute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 25

    else:

        return Age
training_data['Age'] = training_data[['Age','Pclass']].apply(compute_age,axis=1)
#gives birdeye view of columns which might have null values

sns.heatmap(training_data.isnull(),yticklabels=False,cbar=False)
training_data.drop('Cabin',axis=1,inplace=True)
training_data.isnull().values.any()
null_columns=training_data.columns[training_data.isnull().any()]

training_data[null_columns].isnull().sum()
training_data.dropna(inplace=True)
training_data.isnull().values.any()
def categorise_sex(cols):

    age = cols[0]

    sex = cols[1]

    

    if age<16:

        return 'child'

    else:

        return cols[1]
training_data['Sex'] = training_data[['Age','Sex']].apply(categorise_sex,axis=1)
sns.countplot(x='Survived',data=training_data,hue='Sex')
sns.countplot(x='Survived',hue='Pclass',data=training_data)
def is_alone(cols):

    siblings_or_spouse = cols[0]

    parents_or_child = cols[1]

    if (siblings_or_spouse == 0) & (parents_or_child == 0):

        return 1

    else:

        return 0



training_data['Is_Alone'] = training_data[['SibSp','Parch']].apply(is_alone,axis=1)
training_data.head()
sns.countplot(x='Survived',hue='Is_Alone',data=training_data)
training_data.info()
training_data['Pclass'] = training_data['Pclass'].astype('object')

training_data['Is_Alone'] = training_data['Is_Alone'].astype('object')
embark = pd.get_dummies(training_data['Embarked'],drop_first=True)

sex = pd.get_dummies(training_data['Sex'],drop_first=True)

pclass = pd.get_dummies(training_data['Pclass'],drop_first=True)
training_data = pd.concat([training_data,sex,embark,pclass],axis=1)
training_data.head()
training_data.drop(['Sex','Embarked','Name','Ticket','Pclass','SibSp','Parch'],axis=1,inplace=True)
training_data.head()
training_data.drop('PassengerId',axis=1,inplace=True)
X = training_data.drop('Survived',axis=1)

y = training_data['Survived']



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)



from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
pred = logmodel.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))
#tuning RFClassifier to get best results



from sklearn.model_selection import GridSearchCV



n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(10, 30, 3),

              'min_samples_leaf': [3,4,5,6] }



# instantiate the model

rf = RandomForestClassifier(random_state=42)



rf = GridSearchCV(rf, param_grid=parameters,

                  cv=n_folds, 

                 scoring="accuracy")



rf.fit(X_train,y_train)



print('\n'+'Enter the best parameters: ',rf.best_params_)



rf_tuned = RandomForestClassifier(bootstrap=True,

                             max_depth=rf.best_params_['max_depth'],

                             min_samples_leaf=rf.best_params_['min_samples_leaf'],

                             n_estimators=100,

                             random_state=42)



rf_tuned.fit(X_train,y_train)



rf_tuned_pred = rf_tuned.predict(X_test)



print(classification_report(y_test,rf_tuned_pred))
#Using SVM



from sklearn.svm import SVC



model = SVC()



model.fit(X_train,y_train)



SVM_predictions = model.predict(X_test)



print(classification_report(y_test,SVM_predictions))
#tuning SVM to get best results

param_grid = {'C':[0.1,1,10,100,1000,10000,100000],'gamma':[1,.1,.01,.001,.0001,.00001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)



grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

print(classification_report(y_test,grid_predictions))
testing_data = pd.read_csv('../input/test.csv')
testing_data.head()
testing_data['Age'] = testing_data[['Age','Pclass']].apply(compute_age,axis=1)
testing_data.head()
testing_data['Sex'] = testing_data[['Age','Sex']].apply(categorise_sex,axis=1)
testing_data['Is_Alone'] = testing_data[['SibSp','Parch']].apply(is_alone,axis=1)
test_data = testing_data[['Pclass','Sex','Age','Fare','Embarked', 'Is_Alone']]
test_data.head()
#gives birdeye view of columns which might have null values

sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False)
test_data.isnull().values.any()
null_columns=test_data.columns[test_data.isnull().any()]

test_data[null_columns].isnull().sum()
ax = sns.boxplot(x='Pclass',y='Fare',data=test_data)

ax.set_ylim(0,100)
def compute_fare(cols):

    Fare = cols[1]

    Pclass = cols[0]

    if pd.isnull(Fare):

        if Pclass == 1:

            return 60

        elif Pclass == 2:

            return 18

        else:

            return 15

    else:

        return Fare
test_data['Fare'] = test_data[['Pclass','Fare']].apply(compute_fare,axis=1)
test_data.isnull().values.any()
test_data['Pclass'] = test_data['Pclass'].astype('object')

test_data['Is_Alone'] = test_data['Is_Alone'].astype('object')
test_embark = pd.get_dummies(test_data['Embarked'],drop_first=True)

test_sex = pd.get_dummies(test_data['Sex'],drop_first=True)

test_pclass = pd.get_dummies(test_data['Pclass'],drop_first=True)
test_data = pd.concat([test_data,test_embark,test_pclass,test_sex],axis=1)
test_data.head()
test_data.drop(['Sex','Pclass','Embarked'],axis=1,inplace=True)
test_data.head()
predictions = rf_tuned.predict(test_data)
predictions = pd.Series(predictions)
result = pd.concat([testing_data['PassengerId'],predictions],axis=1)
result.columns = ['PassengerId','Survived']
result.head()
filename = 'Titanic Predictions - RF_TUNED.csv'



result.to_csv(filename,index=False)



print('Saved file: ' + filename)