# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head() #to get the sneak peak of the data
train.describe()
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data = train,palette = 'winter')
train.groupby('Pclass').mean()['Age']
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.describe()
train = train.drop(['Ticket','Cabin'],axis = 1)
train.head()
survived_sex = train[train['Survived']==1]['Sex'].value_counts()
dead_sex = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
figure = plt.figure(figsize=(13,8))
plt.hist([train[train['Survived']==1]['Age'],train[train['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
figure = plt.figure(figsize=(13,8))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
survived_embarked = train[train['Survived']==1]['Embarked'].value_counts()
dead_embarked = train[train['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked,dead_embarked])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))
plt.figure(figsize=(13,8))
ax = plt.subplot()
ax.scatter(train[train['Survived']==1]['Age'],train[train['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(train[train['Survived']==0]['Age'],train[train['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
test.describe()
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test.describe()
# extracting and then removing the targets from the training data
targets = train.Survived
train.drop('Survived',1,inplace=True)

#merging train data and test data for future feature engineering
titanic = train.append(test)
titanic.reset_index(inplace=True)
titanic.drop('index',inplace=True,axis=1)
titanic.head()
titanic.describe()
age_fare = pd.DataFrame()

age_fare['Age'] = titanic['Age']
age_fare['Fare'] = titanic['Fare']

age_fare.head()
embarked = pd.get_dummies( titanic.Embarked , prefix='Embarked' )
embarked.head()
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = titanic[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
title.head()
PClass = pd.get_dummies(titanic.Pclass, prefix = 'Pclass')
PClass.head()
sex = pd.Series( np.where( titanic.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
sex.head()
family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = titanic[ 'Parch' ] + titanic[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()
combined = pd.concat( [ age_fare, title, PClass, sex, embarked, family] , axis=1 )
combined.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
titanic_train = combined[:891]
titanic_test = combined[891:]
titanic_test.describe()
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(titanic_train, targets)
features = pd.DataFrame()
features['feature'] = titanic_train.columns
features['importance'] = clf.feature_importances_

features.sort_values(['importance'],ascending=False)
def ML_Algol(x):
    algorithm = x
    algorithm.fit(titanic_train,targets)
    print (algorithm.score( titanic_train , targets ))
model = RandomForestClassifier(n_estimators=100)
ML_Algol(model)
model = SVC()
ML_Algol(model)
model = KNeighborsClassifier(n_neighbors = 3)
ML_Algol(model)
model = GaussianNB()
ML_Algol(model)
model = LogisticRegression()
ML_Algol(model)
test_Y = model.predict(titanic_test)
passenger_id = titanic[891:].PassengerId
test_new = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test_new.shape
test_new.head()
test_new.to_csv( 'titanic_pred.csv' , index = False )
