import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from time import time

import joblib

import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,GradientBoostingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)
train_csv = pd.read_csv('/kaggle/input/titanic/train.csv')

test_csv = pd.read_csv('/kaggle/input/titanic/test.csv')
train_csv.head()
test_csv.head()
test_csv['Survived'] = -1
test_csv.head()
print(train_csv.shape[1])

print(test_csv.shape[1])
titanic_df = pd.concat([train_csv,test_csv], ignore_index=True, sort = True)
# html = titanic_df.to_html()

# with open('data_1.html', 'w') as f:

#     f.write(html)
titanic_df.shape
titanic_df.head()
titanic_df.isnull().sum()
titanic_df.drop(columns=['Cabin'], inplace=True)
titanic_df.head()
titanic_df.isnull().sum()
titanic_df['Age'].value_counts()
age_df = train_csv[['Age','Survived']]
age_gb = age_df.groupby('Age').agg(pd.Series.mode)
h =  age_gb.to_html()

with open('t.html','w') as f:

    f.write(h)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(titanic_df[['Age']])

titanic_df['Age'] = imputer.transform(titanic_df[['Age']])
titanic_df.isnull().sum()
titanic_df.head()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(titanic_df[['Fare']])

titanic_df['Fare'] = imputer.transform(titanic_df[['Fare']])
titanic_df.isnull().sum()
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer.fit(titanic_df[['Embarked']])

titanic_df['Embarked'] = imputer.transform(titanic_df[['Embarked']])
titanic_df.isnull().sum()
titanic_df.head()
sns.catplot(x='SibSp', y='Survived', data=train_csv, kind='point', aspect=2)
sns.catplot(x='Parch', y='Survived', data=train_csv, kind='point', aspect=2)
titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df.head()
titanic_df.drop(['SibSp','Parch'],axis=1, inplace=True)
titanic_df.head()
enc = OrdinalEncoder()

sex = [['male',0],['female',1]]

enc.fit(sex)

titanic_df[['Sex']] = enc.transform(titanic_df[['Sex']])
titanic_df.head()
enc = OneHotEncoder(sparse = False)

Embarked_encode = enc.fit_transform(titanic_df[['Embarked']])
df_embarked = pd.DataFrame(data=Embarked_encode, columns=['E_0','E_1','E_2'])
df_embarked.head()
titanic_df = pd.concat([titanic_df,df_embarked], axis=1)
titanic_df.head()
titanic_df.drop(['Ticket','Name'], axis=1,inplace = True)
titanic_df.head()
titanic_df.drop(['Embarked'], axis=1,inplace = True)
titanic_df.head()
train_data = titanic_df[titanic_df['Survived'] != -1]

test_data = titanic_df[titanic_df['Survived'] == -1]
print(train_data.shape)

print(test_data.shape)
test_data.drop(['Survived'],axis=1, inplace=True)
test_data.shape
features = train_data.drop(['Survived'], axis=1)

labels = train_data['Survived']
X_train,X_test, y_train,y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)
X_val,X_test, y_val, y_test = train_test_split(X_test,y_test, test_size=0.5, random_state = 42)
def print_results(results):

    print(f'Best params: {results.best_params_} \n')

    

    means = results.cv_results_['mean_test_score']

    stds =  results.cv_results_['std_test_score']

    

    for mean,std,params in zip(means,stds, results.cv_results_['params']):

        print(f'{round(mean,3)}(+/-{round(std,3)}) for {params}')
clf = LogisticRegression()

parameters = {

    

    'C': [0.001,0.01, 0.1, 1, 10, 100, 1000]

}



cv = GridSearchCV(clf,parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'LR_model.pkl')
clf = RidgeClassifier()

parameters = {

    'alpha': [0.01,0.1,1,10,100,1000],

    'normalize' : [True, False]

}

cv = GridSearchCV(clf,parameters,cv=5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'RC_model.pkl')
clf = DecisionTreeClassifier()

parameters = {

    'max_depth' : [5,10,15,50,100,None],

}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'DC_model.pkl')
clf = KNeighborsClassifier()

parameters = {

    'n_neighbors' : [3,5,7,10,12],

    'metric' : ['euclidean', 'manhattan' , 'chebyshev']



}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'KNN_model.pkl')
clf = RandomForestClassifier()

parameters = {

    'n_estimators' : [5, 50, 250],

    'max_depth' : [2, 4, 8, 16, 62, None]

}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'RFC_model.pkl')
clf = AdaBoostClassifier()

parameters = {

    'n_estimators' : [5, 50, 250],

}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'ABC_model.pkl')
clf = BaggingClassifier()

parameters = {

    'n_estimators' : [5, 50, 100, 250],

}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'BC_model.pkl')
clf = GradientBoostingClassifier()

parameters = {

    'n_estimators' : [5, 50, 250],

    'max_depth' : [1, 3, 5, 7, 9],

    'learning_rate' : [0.01, 0.1, 1, 10, 100]

}

cv = GridSearchCV(clf, parameters, cv = 5)

cv.fit(X_train,y_train)

print_results(cv)
joblib.dump(cv.best_estimator_,'GBC_model.pkl')
models = {}



for mdl in ['LR', 'RC', 'DC', 'KNN','RFC', 'ABC', 'BC' , 'GBC']:

    models[mdl] = joblib.load(f'{mdl}_model.pkl')
models
def evaluate_model(name, model, features, labels):

    start = time()

    pred = model.predict(features)

    end = time()

    accuracy = round(accuracy_score(labels,pred),3)

    precision = round(precision_score(labels,pred),3)

    recall = round(recall_score(labels,pred),3)

    print(f'{name} === Accuracy : {accuracy} , Precision : {precision} , Recall : {recall} , Latency : {round((end-start)*1000,1)}ms')
for name, mdl in models.items():

    evaluate_model(name, mdl, X_val, y_val)
for name, mdl in models.items():

    evaluate_model(name, mdl, X_test, y_test)
bagginclassifier = models['BC']

bagginclassifier.fit(features,labels)

pred  = bagginclassifier.predict(test_data)

test_data_pid = test_data['PassengerId']

results = pd.Series(data = pred, name = 'Survived', dtype='int64')

df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})

df.to_csv("submission_BC.csv", index=False, header=True)
GradientBoostingClassifier = models['GBC']

GradientBoostingClassifier.fit(features,labels)

pred  = GradientBoostingClassifier.predict(test_data)

test_data_pid = test_data['PassengerId']

results = pd.Series(data = pred, name = 'Survived', dtype='int64')

df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})

df.to_csv("submission_GBC.csv", index=False, header=True)
RandomForestClassifier = models['RFC']

RandomForestClassifier.fit(features,labels)

pred  = RandomForestClassifier.predict(test_data)

test_data_pid = test_data['PassengerId']

results = pd.Series(data = pred, name = 'Survived', dtype='int64')

df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})

df.to_csv("submission_RFC.csv", index=False, header=True)
DecisionTreeClassifier = models['DC']

DecisionTreeClassifier.fit(features,labels)

pred  = DecisionTreeClassifier.predict(test_data)

test_data_pid = test_data['PassengerId']

results = pd.Series(data = pred, name = 'Survived', dtype='int64')

df = pd.DataFrame({"PassengerId":test_data_pid, "Survived":results})

df.to_csv("submission_DC.csv", index=False, header=True)