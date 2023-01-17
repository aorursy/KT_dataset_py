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
# Loading .csv files

df_tr = pd.read_csv('/kaggle/input/titanic/train.csv').rename(columns={'Survived':'Target'})
df_tr.head()
# Preprocessing data

def sex_sort(x):
    if x == 'male':
        return 1
    elif x == 'female':
        return 0
    else:
        return 0.5

def embarked_sort(x):
    if x == 'C':
        return 0
    elif x == 'Q':
        return 0.5
    elif x == 'S':
        return 1
    else:
        return -2

def is_cherbourg(x):
    if x == 'C':
        return 1
    else:
        return 0
    
def is_queenstown(x):
    if x == 'Q':
        return 1
    else:
        return 0
    
def is_southampton(x):
    if x == 'S':
        return 1
    else:
        return 0

def pre_process(df):
    # Sorting Sex
    df['numeric_sex'] = df['Sex'].apply(lambda x: sex_sort(x))
    df['Sex'] = df['numeric_sex']
    
    df['Queenstown'] = df['Embarked'].apply(lambda x: is_queenstown(x))
    df['Cherbourg'] = df['Embarked'].apply(lambda x: is_cherbourg(x))
    df['Southampton'] = df['Embarked'].apply(lambda x: is_southampton(x))
    
    return df

df_train = pre_process(df_tr).set_index('PassengerId')



# removing NaNs
df_train = df_train.fillna(-999)

df_train.head()
from sklearn.model_selection import train_test_split
# Function to select features and split into training data into x and y
def feature_split(features,df):
    X = df[features]
    y = df[['Target']]
    return X, y

# Loading and processing testing data
def load_test_data(features,scaler):
    df_X = pd.read_csv('/kaggle/input/titanic/test.csv')
    df_X = pre_process(df_X).set_index('PassengerId').fillna(-999)
    X = df_X[features]
    X_scaled = scaler.transform(X)
    
    return X_scaled, df_X.index

# Fucntion to cross validate model

from sklearn.model_selection import cross_val_score

def cross_val_model(classifier, X, y, cv=10, classifier_name='Classifier'):
    
    # Finding Cross Validation Score
    cross_val = cross_val_score(RandomForestCLF, X, y.values.ravel(), cv=cv)
    string = f'{classifier_name} Accuracy: {cross_val.mean()} (+/- {cross_val.std()*2})'
    return string
    

from sklearn.ensemble import RandomForestClassifier

# Random Forest Features
features_rf = ['Pclass','Sex','Age','SibSp','Parch','Fare','Queenstown','Cherbourg','Southampton']

# Random Forest Object
RandomForestCLF = RandomForestClassifier(n_estimators=200)

# Feature and Splitting
X, y = feature_split(features_rf,df_train)

print(cross_val_model(RandomForestCLF, X, y, 10, 'Random Forest'))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# KNN Features
features_knn = ['Pclass','Sex','Age','SibSp','Parch','Fare','Queenstown','Cherbourg','Southampton']


for N in [1,2,3,4,5,6,7,8,9,10,11]:   
    # KNN Object
    knnCLF = KNeighborsClassifier(n_neighbors = N)
    
    # Feature and Splitting
    X, y = feature_split(features_knn,df_train)
    
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    
    print(cross_val_model(knnCLF, X, y, 10, f'{N}-NN'))

from sklearn.svm import SVC


# SVC Features
features_svc = ['Pclass','Sex','Age','SibSp','Parch','Fare','Queenstown','Cherbourg','Southampton']

kernels = ['rbf','poly']

for kernel in kernels:
    svcCLF = SVC(kernel=kernel,probability=True)
    
    # Feature and Splitting
    X, y = feature_split(features_svc,df_train)
    
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    
    print(cross_val_model(svcCLF, X, y, 10, f'{kernel} SVC'))
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Queenstown','Cherbourg','Southampton']

# Respective classifier objects
RandomForestCLF = RandomForestClassifier(n_estimators=500)
svcCLF = SVC(kernel='poly',probability=True)
knnCLF = KNeighborsClassifier(n_neighbors = 101)

votingCLF = VotingClassifier(estimators=[
     ('svc',svcCLF),('rf',RandomForestCLF),('knn',knnCLF)], voting='soft')

# Feature and Splitting
X, y = feature_split(features,df_train)

scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

print(cross_val_model(votingCLF, X, y, 10, f'Voting Classifier'))
# Fitting Voting classifier with full X,y data
votingCLF.fit(X,y.values.ravel())


X_to_predict, df_test_index = load_test_data(features,scaler)
predictions = votingCLF.predict(X_to_predict)
submission = pd.DataFrame({"PassengerId": df_test_index,"Survived": predictions})
submission
submission.to_csv('/kaggle/working/submission.csv',index=False)
