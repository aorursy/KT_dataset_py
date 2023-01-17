# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV

# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")



full = train.append( test , ignore_index = True )

titanic = full[ :891 ]



del train , test



print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
full.head(10)
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked.head()
# Create a new variable for every unique value of Pclass

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
# Fill missing values of Age with the average of Age (mean)

age = pd.DataFrame()

age['Age'] = full.Age.fillna( full.Age.mean() )



# Fill missing values of Fare with the average of Fare (mean)

fare = pd.DataFrame()

fare[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )



# Transform Sex into binary values 0 and 1

sex = pd.DataFrame()

sex['Sex'] = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )



full.head(10)
titanic['Sex'].unique()
titanic['Sex'].value_counts()
full.isnull().any()
full['Embarked'].isnull().value_counts()
# Select which features/variables to include in the dataset from the list below:

# imputed , embarked , pclass , sex , family , cabin , ticket



full_X = pd.concat( [  embarked , sex, age, fare, pclass, full['SibSp'], full['Parch']] , axis=1 )

full_X.head()
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_X[ 0:891 ]

train_valid_y = titanic.Survived

test_X = full_X[ 891: ]

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
#model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
# Score the model

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )

rfecv.fit( train_X , train_y )
print (rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y ))

print( "Optimal number of features : %d" % rfecv.n_features_ )
test_Y = model.predict( test_X )

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

print(test.shape, test.head())

#test.to_csv( 'titanic_pred.csv' , index = False )
test_Y = model.predict( test_X )

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test['Survived'] = test['Survived'].astype(int)

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )
test