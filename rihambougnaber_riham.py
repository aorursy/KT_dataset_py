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

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler







train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

print(train_df.columns.values)

print(test_df.columns.values)
train_df.head(20)
train_df.tail()
train_df.isna()
train_df.isna().sum()/train_df.shape[0]
train_df.isnull().sum()
train_df.info()

test_df.info()

train_df.describe(percentiles=[0.05,.1,.2,.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98,0.99])

train_df[train_df['Ticket']=='347082']
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[train_df['Age']<1]
g = sns.FacetGrid(train_df, col='Survived',height=6)

g.map(plt.hist, 'Age', bins=200)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=3.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
g = sns.FacetGrid(train_df, col='Survived',row='Sex',height=4,xlim=(0,85),ylim=(0,100))

g.map(plt.hist, 'Age', bins=10, color="g").set_axis_labels("Age of the passanger", "Passanger Nuumber")
grid = sns.FacetGrid(train_df, row='Embarked', height=4.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep',)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=4.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.8, ci=None)

grid.add_legend()
train_df.corr()
train_df.corr().style.background_gradient(cmap='Reds')
X = train_df[['Fare']].copy()

y = train_df['Survived'].copy() # or df.Survived.values

y.count() #so we need no preprocessing
model1 = DecisionTreeClassifier()



model1.fit(X,y)



p = model1.predict(X)



np.sum(y==p)/len(y)
from sklearn.metrics import accuracy_score

accuracy_score(y,p)
train_set = train_df[0:700].copy()

test_set = train_df[700:].copy()

print(train_set.shape,test_set.shape)



X_train = train_set[['Fare']]

y_train = train_set['Survived']



X_test = test_set[['Fare']]

y_test = test_set['Survived'] 



#First Step

model2 = DecisionTreeClassifier()

#Second step

model2.fit(X_train,y_train)



#Prediction

pred_train = model2.predict(X_train)

pred_test = model2.predict(X_test)



print('Train Score :',1-accuracy_score(y_train,pred_train))

print('Test Score :',1-accuracy_score(y_test,pred_test))

from sklearn.model_selection import KFold, StratifiedKFold





data =  np.array( [[1,2,3,4,5],[8,5,2,1,5],[1,0,0,1,0]] ).transpose()

example_df =  pd.DataFrame(data, columns=['x1','x2','target'])

example_df.shape
example_df.describe()
example_df.target.hist()
train_df.Survived.hist()
cv =  StratifiedKFold(n_splits=2,random_state=0,shuffle=True)

y = example_df.target.values

X = example_df[['x1','x2']]

for train_index, test_index in cv.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)
example_df.loc[train_index]


df = pd.read_csv('../input/titanic/train.csv')

X = df[['Fare','SibSp','Parch','Age']]

y = df['Survived']



# Preprocessing

X['Age'].fillna(-99,inplace=True)



#Cross-validation

cv = StratifiedKFold(n_splits=2,random_state=90,shuffle=True)



#Train 

for train_index, test_index in cv.split(X, y):

    X_train = X.loc[train_index]

    y_train = y[train_index]

    

    X_test = X.loc[test_index]

    y_test = y[test_index]

    

    

    model = DecisionTreeClassifier()

    model.fit(X_train,y_train)

    

    pred_train = model.predict(X_train)

    pred_test = model.predict(X_test)

    

    score = 1 - accuracy_score(y_train,pred_train)

    print("TRAIN error:", score)

    

    

    score = 1 - accuracy_score(y_test,pred_test)

    print("Test error:", score)

    

    print ('Fold','*'*50)

    

from sklearn.preprocessing import OrdinalEncoder



df = pd.read_csv('../input/titanic/train.csv')

X = df[['Fare','SibSp','Parch','Sex']]

y = df['Survived']



# Preprocessing



'''

first solution for replacing catagorical value

X.loc[ X['Sex']=='male', 'Sex' ] = 0

X.loc[ X['Sex']=='female', 'Sex'] = 1



'''



lb = OrdinalEncoder()

X['Sex'] = lb.fit_transform(X[['Sex']]).astype(int)









#Cross-validation

cv = StratifiedKFold(n_splits=5,random_state=90,shuffle=True)



results_df = pd.DataFrame(data=np.zeros((5,2)),columns=['Train_error', 'Test_error'])

fold=0



#Train 

for train_index, test_index in cv.split(X, y):

    X_train = X.loc[train_index]

    y_train = y[train_index]

    

    X_test = X.loc[test_index]

    y_test = y[test_index]

    

    

    model = DecisionTreeClassifier()

    model.fit(X_train,y_train)

    

    pred_train = model.predict(X_train)

    pred_test = model.predict(X_test)

    

    score = 1 - accuracy_score(y_train,pred_train)

    print("TRAIN:", score)



    results_df.loc[fold,'Train_error'] = round(score*100,2)

    

    score = 1 - accuracy_score(y_test,pred_test)

    print("Test:", score)

    

    results_df.loc[fold,'Test_error'] = round(score*100,2)

    

    print ('Fold','*'*50)

    fold +=1
results_df.describe()
sub = test_df[['Fare','SibSp','Parch','Sex']]

lb = OrdinalEncoder()

sub['Sex'] = lb.fit_transform(sub[['Sex']]).astype(int)



sub['Fare'].fillna(sub['Fare'].mean(), inplace=True)



cv = StratifiedKFold(n_splits=5,random_state=10,shuffle=True)



results_df = pd.DataFrame()

fold=0



for train_index, test_index in cv.split(X, y):

    X_train = X.loc[train_index]

    y_train = y[train_index]



    X_test = X.loc[test_index]

    y_test = y[test_index]



    model = DecisionTreeClassifier(max_depth=7, min_samples_leaf=2)

    model.fit( X_train,  y_train)



    pred_sub   = model.predict_proba(sub)[:,1]

   

    results_df['fold_'+str(fold)] = pred_sub

    

    fold +=1
results_df
#Mean strategy

preds = (results_df.mean(axis=1) >=0.5).astype(int)

preds
#Majority voting strategy

preds = (results_df.mean(axis=1) >=0.5).astype(int)
my_final_sub = pd.read_csv('../input/titanic/test.csv')[['PassengerId']]

my_final_sub['Survived'] = preds



my_final_sub.to_csv('submission.csv', index=False)

my_final_sub