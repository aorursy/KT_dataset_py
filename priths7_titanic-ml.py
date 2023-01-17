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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.isnull().sum()
train.head(10)
import matplotlib.pyplot as plt

%matplotlib inline
# Function to plot bar graph 

def bar_plot(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
train_test_comb = [train,test]

# function to seperate titles from name

for data in train_test_comb:

    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    

train['Title'].value_counts()
titles = {"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":4,"Rev":4,"Col":4,"Major":4,"Mlle":4,"Lady":4,"Capt":4,"Ms":4,"Jonkheer":4,"Sir":4,"Countess":4,"Don":4,"Mme":4}

for datas in train_test_comb:

    datas['Title'] = datas['Title'].map(titles)
train = train.drop(['Name','Ticket'],axis=1)

test = test.drop(['Name','Ticket'],axis=1)
train['FamSize'] = train['SibSp'] + train['Parch'] + 1

test['FamSize'] = test['SibSp'] + test['Parch'] + 1
train = train.drop(['SibSp','Parch'],axis=1)

test = test.drop(['SibSp','Parch'],axis=1)
temp_train = train.drop(['Cabin'],axis=1)

temp_test = test.drop(['Cabin'],axis=1)
temp_train["Fare"] = temp_train["Fare"].apply(pd.to_numeric, errors='coerce')

temp_test["Fare"] = temp_test["Fare"].apply(pd.to_numeric, errors='coerce')

temp_train["Fare"].fillna(temp_train["Fare"].median())

temp_test["Fare"].fillna(temp_test["Fare"].median())
temp_train.fillna(0,inplace=True)

temp_test.fillna(0,inplace=True)
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

temp_train['Sex'] = label_encoder.fit_transform(temp_train['Sex'])

temp_train['Embarked'] = label_encoder.fit_transform(temp_train['Embarked'].astype(str))

temp_test['Sex'] = label_encoder.fit_transform(temp_test['Sex'])

temp_test['Embarked'] = label_encoder.fit_transform(temp_test['Embarked'].astype(str))
X = temp_train.drop(['Survived'],axis=1)

Y = temp_train.Survived                                                                                            
#importing library and building model

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn import model_selection

#model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

models_set = [

    SVC(),

    RandomForestClassifier(),

    LogisticRegression(),

    KNeighborsClassifier(),

    GaussianNB(),

    DecisionTreeClassifier(),

    LinearDiscriminantAnalysis()

    ]



MLA_res = pd.DataFrame(columns=['MLA_name','MLA_Accuracy_Mean'])

index = 0

for algo in models_set:

    MLA_name = algo.__class__.__name__

    cv_res = model_selection.cross_val_score(algo,X,Y,cv = 5)

    MLA_res.loc[index, 'MLA_name'] = MLA_name

    MLA_res.loc[index, 'MLA_Accuracy_Mean'] = cv_res.mean()

    index = index + 1

    

MLA_res.sort_values(by = ['MLA_Accuracy_Mean'], ascending = False, inplace = True)
MLA_res
model = RandomForestClassifier(n_estimators=13)

model.fit(X,Y)

val_pred = model.predict(temp_test)



print(val_pred)

my_submission = pd.DataFrame({'PassengerId': temp_test.PassengerId, 'Survived': val_pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)