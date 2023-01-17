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
train.head()
train.describe()
train.info()
train.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace = True)

test['Embarked'].fillna('S',inplace = True)

train['Age'].fillna(train['Age'].mean(),inplace = True)

test['Age'].fillna(test['Age'].mean(),inplace = True)

print(train.head())

print(train.info())

print(train['Embarked'].value_counts)

print(train.isnull().sum())
a = {'male' : 0 , 'female' : 1}

e = {'S' : 0 , 'Q' : 1 , 'C' : 1 }

train.replace({'Sex' : a , 'Embarked' : e},inplace=True)

#test.replace({'Sex' : a , 'Embarked' : e},inplace=True)

print(test.head())

train.head()
import seaborn as sns

t = train.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)

c = abs(t.corr())

sns.heatmap(c,annot = True)
import seaborn as sns

t = train.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)

c = (t.corr())

y = train['Survived']

sns.heatmap(c,annot = True)
sns.countplot(x = 'Survived', data = train)
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
train['Fare'].hist(bins = 50)
sns.countplot(x = 'Survived', hue = 'SibSp', data = train)
sns.countplot(x = 'Survived', hue = 'Parch', data = train)
#Import models from scikit learn module: 

from sklearn.linear_model import LogisticRegression   

from sklearn.model_selection import KFold 

from sklearn.ensemble import RandomForestClassifier 

from sklearn import metrics
def classification_model(model, data, predictors, outcome):  

    #Fit the model:  

    model.fit(data[predictors],data[outcome])    

    #Make predictions on training set:  

    predictions = model.predict(data[predictors])    

    #Print accuracy  

    accuracy = metrics.accuracy_score(predictions,data[outcome])  

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds  

    kf = KFold(5,shuffle=True)  

    error = []  

    for train, test in kf.split(data):

        # Filter training data    

        train_predictors = (data[predictors].iloc[train,:])        

        # The target we're using to train the algorithm.    

        train_target = data[outcome].iloc[train]        

        # Training the algorithm using the predictors and target.    

        model.fit(train_predictors, train_target)

        #Record error from each cross-validation run    

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

     

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 

    # %s is placeholder for data from format, next % is used to conert it into percentage

    #.3% is no. of decimals

    return model
a = {'male' : 0 , 'female' : 1}

e = {'S' : 0 , 'Q' : 1 , 'C' : 1 }

test.replace({'Sex' : a , 'Embarked' : e},inplace=True)

test.head()


test['Age'].fillna(test['Age'].median(),inplace = True)

test['Fare'].fillna(test['Fare'].median(),inplace = True)

t = test.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)

t
output = 'Survived'

model = RandomForestClassifier()

predict = ['Sex','Parch','SibSp','Fare','Age','Embarked']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(t[predict])

a

#'Age','Parch','SibSp',
my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': a})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
test.info()

test.describe()

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV


output = 'Survived'

model = RandomForestClassifier()

predict = ['Sex','Parch','SibSp','Fare','Age','Embarked']

param =  {  'n_estimators': [200, 500],    'max_features': ['auto', 'sqrt', 'log2'],    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

mod  = GridSearchCV(estimator=model, param_grid=param, cv= 5)

mod.fit(train[predict], train[output])

mod.best_params_
output = 'Survived'

model = RandomForestClassifier(criterion= 'entropy', max_depth= 5,max_features= 'auto',n_estimators= 500)

predict = ['Sex','Parch','SibSp','Fare','Age','Embarked','Pclass']

classification_model(model,train,predict,output)

m = classification_model(model,train,predict,output)

a = m.predict(t[predict])

a