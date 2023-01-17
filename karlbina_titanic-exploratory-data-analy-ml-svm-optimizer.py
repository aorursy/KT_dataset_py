"""
Create by karl bina
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_rows', None)
Test = pd.read_csv('/kaggle/input/titanic/test.csv')
Train = pd.read_csv('/kaggle/input/titanic/train.csv')
Train_set = Train.copy()
Test_set = Test.copy()
Train_set.head()
Test_set.head()
Train_set.shape
Train_set.dtypes.value_counts()
Train_set.dtypes.value_counts().plot.bar()
import seaborn  as sns
############################ identify empty values #######################################
plt.figure(figsize=(12, 8))
sns.heatmap(Train_set.isna(), cbar=False)
Train_set.isna().sum()
### Percentage of empty data
(Train_set.isna().sum() /Train_set.shape[0]).sort_values(ascending=False)
Train_set['Survived'].value_counts(normalize=True)
for value in Train_set.select_dtypes('float64'):
    plt.figure()
    sns.distplot(Train_set[value])
for value in Train_set.select_dtypes('int64'):
    plt.figure()
    try:
        sns.distplot(Train_set[value])
    except RuntimeError as re:
        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
            sns.distplot(Train_set[value], kde_kws={'bw': 0.1})
        else:
            raise re
plt.figure(figsize=(12, 8))
sns.countplot(x='Sex', hue='Survived', data=Train_set)
Survived = Train_set[Train_set['Survived'] == 1].drop(['Survived','PassengerId','Name','Sex',
                                                       'Cabin','Ticket','Embarked'],
                                                      axis=1)
Die = Train_set[Train_set['Survived'] == 0].drop(['Survived','PassengerId','Name','Sex',
                                                       'Cabin','Ticket','Embarked'],
                                                      axis=1)
df = Train_set.drop(['Survived','PassengerId','Name','Sex',
                                                       'Cabin','Ticket','Embarked'],
                                                      axis=1)
for col in df:
    try:
        plt.figure()
        sns.distplot(Survived[col], label='Survived')
        sns.distplot(Die[col], label='Die')
        plt.legend()
    except RuntimeError as re:
        if str(re).startswith("Selected KDE bandwidth is 0. Cannot estimate density."):
            plt.figure()
            sns.distplot(Survived[col], label='Survived', kde_kws={'bw': 0.1})
            sns.distplot(Die[col], label='Die', kde_kws={'bw': 0.1})
            plt.legend()
        else:
            raise re    
Train_set.drop(['Survived','PassengerId','Name','Cabin','Ticket','Embarked'],
                                                      axis=1).groupby(['Sex','Pclass']).mean()
sns.pairplot(Train_set)
Train_set['Sex'].value_counts()
df = Train.copy() 
df.head()
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
def Encodage(df):
    code = {'male':1,
            'female':0}
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
        
    return df
def Imputation(df):
    df = df.drop(['PassengerId','Name','Cabin','Ticket','Embarked'], axis=1)
    df = df.dropna(axis=0)
    
    return df
def pre_processing(df):
    df = Imputation(df)
    df = Encodage(df)
    
    X = df.drop(['Survived'], axis=1)
    y = df["Survived"]
    
    return X, y
X_train, y_train = pre_processing(df)
print(y_train.shape)
print(X_train.shape)
def Imputation_test(df):
    #df = df.dropna(axis=0)
    id_passenger = df["PassengerId"]
    df = df.drop(['PassengerId','Name','Cabin','Ticket','Embarked'], axis=1)
    
    return df,id_passenger
def pre_processing_test(df):
    df, PassengerId = Imputation_test(df)
    df = Encodage(df)
        
    return df, PassengerId
X_test, PassengerId = pre_processing_test(Test_set)
plt.figure()
sns.heatmap(X_test, cbar=False)
mean_fare = X_test['Fare'].mean()
mean_fare
mean_age = X_test['Age'].mean()
mean_age
X_test['Fare'].fillna(value=X_test['Fare'].median(), inplace=True)
X_test['Age'].fillna(value=X_test['Age'].median(), inplace=True)
from sklearn.svm import  SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
model_list = []
preprocessor = make_pipeline(SelectKBest(f_classif, k='all'))
randomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
adaboost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
svm = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=1))
knn = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
bayes = make_pipeline(preprocessor, StandardScaler(), GaussianNB())
model_list = {'randomForest':randomForest,
             'adaboost': adaboost,
             'svm': svm,
             'knn': knn,
             'bayes': bayes
             }
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
predict_list = {}

def Evaluation(key, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predict_list.update({key: y_pred})
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='accuracy',
                                            train_sizes=np.linspace(0.1,1,10))
    
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.ylabel('Score')
    plt.title(key)
    plt.legend()
    
for key, model in model_list.items():
    Evaluation(key, model)
    
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy as sc
param_svm = [{'svc__C': sc.stats.expon(scale=100), 
                        'svc__gamma': sc.stats.expon(scale=.1),
                        'svc__kernel': ['rbf']},
                       {'svc__C': sc.stats.expon(scale=100), 
                        'svc__kernel': ['linear']}]
                        
grid_random = RandomizedSearchCV(estimator=svm, param_distributions = param_svm, 
                        cv =5, scoring ='accuracy', refit = True, n_jobs = -1,
                       random_state=1, n_iter=100)
grid_random.fit(X_train, y_train)
print(grid_random.best_params_)
y_pred = grid_random.predict(X_test)
grid_random.best_score_
Evaluation('svm',grid_random.best_estimator_)
submission = pd.DataFrame(PassengerId)
submission['Survived'] = y_pred
submission.to_csv('submission.csv', header= True, index= False) 

submission.shape
knn
param_knn = [{'n_neighbors': np.arange(1,50),
              'metric': ['euclidean', 'manhattan']}]
grid_random_knn = RandomizedSearchCV(KNeighborsClassifier(), param_knn,
                                     cv=4, n_jobs = 1,
                                    n_iter=30)
grid_random_knn.fit(X_train, y_train)
print(grid_random_knn.best_params_)
y_pred = grid_random_knn.predict(X_test)
Evaluation('knn',grid_random_knn.best_estimator_)
grid_random_knn.best_score_
