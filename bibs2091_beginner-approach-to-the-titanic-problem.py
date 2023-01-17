# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder 

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('/kaggle/input/titanic/train.csv')
print("data shape: " ,data.shape)

print("data columns: " ,list(data.columns))
data.info()
data.describe()

data['Survived'].value_counts().plot.pie(figsize=(6,6))
def plot_by(data,feature):

    groupedData = data[feature].groupby(data["Survived"])

    groupedData = pd.DataFrame({

        'Survived' : groupedData.get_group(1),

        'Dead': groupedData.get_group(0)

    })

    histogram = groupedData.plot.hist(bins=40,alpha=0.4)

    histogram.set_xlabel(feature)

    histogram.plot()
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

data['Sex'].values.reshape(-1,1)

enc = OneHotEncoder()

ss = pd.DataFrame(enc.fit_transform(data['Sex'].values.reshape(-1,1)).toarray(),columns=['male','female'])

plot_by(data,'Parch')
plot_by(data,'SibSp')
data['Familly'] = data['SibSp'] + data['Parch']

plot_by(data,'Familly')
#new features male and female using one hot encoders 

data['Male'] = ss['male']

data['female'] = ss['female']

data = data.loc[:, data.columns != 'Sex']

data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

data.head()
#new feature (title)

title_names = (data['Title'].value_counts() < 10)

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

l = LabelEncoder()

data['Title'] = l.fit_transform(data['Title'])
Embarked = set(data.loc[data["Embarked"].notnull()]["Embarked"])

Embarked_to_number = {ni: indi for indi, ni in enumerate(set(Embarked))}

print(Embarked)

print(Embarked_to_number)

data['Embarked'] = data['Embarked'].map(Embarked_to_number)
data['alone'] = (data['Familly'] == 0).astype('int')
data['Embarked'].hist()
data.isnull().sum()
data = data.loc[:, data.columns != 'PassengerId']

data = data.loc[:, data.columns != 'Cabin']
print(data.shape)

#complete missing age with median

data['Age'].fillna(data['Age'].median(), inplace = True)



#complete embarked with mode

data['Embarked'].fillna(data['Embarked'].median(), inplace = True)



#complete missing fare with median

data['Fare'].fillna(data['Fare'].median(), inplace = True)

data = data.dropna(how='any')



#####################

data.isnull().sum()

data = data.loc[:, data.columns != 'Ticket']

data = data.loc[:, data.columns != 'Name']

data.info()

import seaborn as sns #the librery we'll use for the job xD



corrmat = data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);
data.head()
data.columns
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix
#preparing the data

X_cols = list(data.columns)

X_cols.remove('Survived')

Y=data['Survived']

rescaledX = StandardScaler().fit_transform(data[X_cols])

X=pd.DataFrame(data = rescaledX, columns= X_cols)

# X=data[X_cols]

X_train,Y_train = X,Y
%%time

svm1 = svm.SVC(kernel='linear',random_state = 42)

svm2 = svm.SVC(kernel='rbf',random_state = 42) 

lr = LogisticRegression(random_state = 42)

gb = GaussianNB()

rf = RandomForestClassifier(random_state = 42)

knn = KNeighborsClassifier(n_neighbors=15)

tree = tree.DecisionTreeClassifier()

models = {"Logistic Regression": lr, 'DecisionTreeClassifier' : tree, "Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,"KNeighborsClassifier": knn ,'GaussianNB': gb}

l=[]

for model in models:

    l.append(make_pipeline(Imputer(),  models[model]))



        

i=0

for Classifier in l:    

    accuracy = cross_val_score(Classifier,X_train,Y_train,scoring='accuracy',cv=5)

    print("===", [*models][i] , "===")

    print("accuracy = ",accuracy)

    print("accuracy.mean = ", accuracy.mean())

    print("accuracy.variance = ", accuracy.var())

    i=i+1

    print("")

    
k_number = [i for i  in range(1,100,2)]

acc = []

ks = []

for i in range(len(k_number)):

    knn = KNeighborsClassifier(n_neighbors=k_number[i])

    accuracy = cross_val_score(knn,X_train,Y_train,scoring='accuracy',cv=5)

    acc.append(accuracy.mean())

    ks.append(k_number[i])

plt.plot(ks,acc)
print("best knn score: ", max(acc))

print("best k is:", ks[acc.index(max(acc))] )
from sklearn.model_selection import GridSearchCV

param_grid = {

    'bootstrap': [True,False],

    'max_depth': [130,140,150,160, None],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 2],

    'min_samples_split': [12, 14, 16],

    'n_estimators': [100, 110, 120, 130,140]

}

# Create a based model

rf = RandomForestClassifier(random_state = 42)

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, Y_train)

print(grid_search.best_score_)

print(grid_search.best_estimator_)

rf = grid_search.best_estimator_
svm2 = svm.SVC(kernel='rbf',random_state = 42) 

svm2.fit(X_train, Y_train)
data= pd.read_csv('/kaggle/input/titanic/test.csv')

test = data.copy()

data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

data['Sex'].values.reshape(-1,1)

enc = OneHotEncoder()

ss = pd.DataFrame(enc.fit_transform(data['Sex'].values.reshape(-1,1)).toarray(),columns=['male','female'])

data['Familly'] = data['SibSp'] + data['Parch']

data['Male'] = ss['male']

data['female'] = ss['female']

data = data.loc[:, data.columns != 'Sex']

data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

title_names = (data['Title'].value_counts() < 10)

data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

l = LabelEncoder()

data['Title'] = l.fit_transform(data['Title'])

Embarked = set(data.loc[data["Embarked"].notnull()]["Embarked"])

Embarked_to_number = {ni: indi for indi, ni in enumerate(set(Embarked))}

data['Embarked'] = data['Embarked'].map(Embarked_to_number)

data['alone'] = (data['Familly'] == 0).astype('int')

data = data.loc[:, data.columns != 'PassengerId']

data = data.loc[:, data.columns != 'Cabin']

data['Age'].fillna(data['Age'].median(), inplace = True)

data['Embarked'].fillna(data['Embarked'].median(), inplace = True)

data['Fare'].fillna(data['Fare'].median(), inplace = True)

data = data.dropna(how='any')

data.isnull().sum()

data = data.loc[:, data.columns != 'Ticket']

data = data.loc[:, data.columns != 'Name']

X_cols = list(data.columns)

rescaledX = StandardScaler().fit_transform(data[X_cols])

X=pd.DataFrame(data = rescaledX, columns= X_cols)

X_test = X
predictions = svm2.predict(X_test)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)