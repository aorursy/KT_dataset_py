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
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

y = train.Survived

passengerid = test.PassengerId



#Creating a whole dataset with train and test dataset 

titanic = train.append(test, ignore_index = True)
#Saving the train and test inde to split later

train_index = len(train)

test_index = len(titanic) - len(test)
titanic.head()
titanic.info()
titanic['Title'] = titanic.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
titanic.Title.value_counts()
normalized_title = {

            'Mr':"Mr",

            'Mrs': "Mrs",

            'Ms': "Mrs",

            'Mme':"Mrs",

            'Mlle':"Miss",

            'Miss':"Miss",

            'Master':"Master",

            'Dr':"Officer",

            'Rev':"Officer",

            'Col':"Officer",

            'Capt':"Officer",

            'Major':"Officer",

            'Lady':"Royalty",

            'Sir':"Royalty",

            'the Countess':"Royalty",

            'Dona':"Royalty",

            'Don':"Royalty",

            'Jonkheer':"Royalty"

            

}
titanic.Title = titanic.Title.map(normalized_title)
print(titanic.Title.value_counts())
grouped = titanic.groupby(['Sex','Title','Pclass'])

print(grouped.Age.median())
titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

titanic.isnull().sum()
#Since fare has only one missing value we will fill it with mean value

titanic.Fare = titanic.Fare.fillna(titanic.Fare.mean())



#For the Cabin since it contains large number of missing values we will fill the unnown values as 'U'

titanic.Cabin = titanic.Cabin.fillna('U')



#For the Embared we will fill it with the most frequent Embarked value

most_Embarked = titanic.Embarked.value_counts().index[0]

titanic.Embarked = titanic.Embarked.fillna(most_Embarked)
titanic.isnull().sum()
#We also the add the member along with the famlily count

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic.FamilySize.head()
titanic['Cabin'].value_counts()
titanic.Cabin = titanic.Cabin.map(lambda x: x[0])

titanic.Cabin.head()
titanic.select_dtypes('object').columns
#Converting the male as 0 and female as 1 using the dictionary and mapping it

titanic.Sex = titanic.Sex.map({"male":0,"female":1})



#Converting the Title ,Cabin, Pclass and Embarked using get_dummies()

title_dummies = pd.get_dummies(titanic.Title , prefix = "Title")

cabin_dummies = pd.get_dummies(titanic.Cabin , prefix = "Cabin")

pclass_dummies = pd.get_dummies(titanic.Pclass , prefix ="Pclass")

embarked_dummies = pd.get_dummies(titanic.Embarked , prefix = "Embarked")
#Concatinating dummy columns with main dataset

titanic_dummies = pd.concat([titanic , title_dummies,cabin_dummies,

                             pclass_dummies,embarked_dummies],axis = 1)
#Dropping the categorical columns in the titanic_dummies

titanic_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'],axis = 1,inplace = True)
titanic_dummies.head()
train_x = titanic_dummies[:train_index]

test_x = titanic_dummies[train_index:]
#Now converting the Survived column as int

train_x.Survived = train_x.Survived.astype(int)
# create X and y for data and target values 

X = train_x.drop('Survived', axis=1).values 

y = train_x.Survived.values
train_x.head()
# create array for test set

X_test = test_x.drop('Survived', axis=1).values
label = train_x.Survived

train_X , val_X , train_Y , val_Y = train_test_split(train_x , label,test_size = 0.2,shuffle = True)
val_X.shape
dtrain_X = train_X.drop('Survived',axis = 1)
dval_X = val_X.drop('Survived',axis = 1)
#Creating parameters for the tuning of model in grid search

params = dict(

            max_depth = [n for n in range(9,15)],

            min_samples_split = [n for n in range(4, 11)], 

            min_samples_leaf = [n for n in range(2, 5)],     

            n_estimators = [n for n in range(10, 60, 10)],

)
#Classifer we are using is RandomForest

model_forest = RandomForestClassifier()
#Building and fitting the model

forest_gs = GridSearchCV(param_grid=params,

                         estimator=model_forest,

                         cv=5)



#forest_gs.fit(X,y)



#Trying to fit the model with train_test_split train data which is train_X

forest_gs.fit(dtrain_X,train_Y)
forest_gs.best_score_
forest_gs.best_estimator_
from sklearn.metrics import mean_absolute_error

p = forest_gs.predict(dval_X)

print(mean_absolute_error(p,val_Y))
prediction_Random_forest = forest_gs.predict(X_test)

prediction_Random_forest
#storing it in submission file

output = pd.DataFrame({"PassengerId":passengerid , "Survived" : prediction_Random_forest})

output.to_csv("Submission5.csv",index = False)
output