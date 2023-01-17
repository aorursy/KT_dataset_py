import pandas as pd

import numpy as np

from ggplot import *

import matplotlib

matplotlib.style.use('ggplot')

%matplotlib inline 
train = pd.read_csv("../input/train.csv")
train.head()
ggplot(train, aes("Sex", fill="factor(Survived)")) + geom_bar()
import seaborn as sns

g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], hue='Survived', palette = 'seismic',size=1.5,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
corr = train.corr()

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(

    corr, 

    cmap = cmap,

    square=True, 

    cbar_kws={ 'shrink' : .9 }, 

    annot = True, 

    annot_kws = { 'fontsize' : 12 }

)
train[["Age", "Survived"]].corr()
sns.boxplot(x="Pclass", y="Age", hue="Survived", data=train)
sns.boxplot(x="Sex", y="Age", hue="Survived", data=train)
sns.jointplot(x="Age", y="Fare", data=train)
train.describe()
facet = sns.FacetGrid( train , hue="Survived" , aspect=4 , row = "Sex")

facet.map( sns.kdeplot , "Fare" , shade= True )

facet.set( xlim=( 0 , train["Fare"].max() ) )

facet.add_legend()
test = pd.read_csv("../input/test.csv")
full = train.append(test)
# Transform Sex into binary values 0 and 1



sex_df = pd.DataFrame()

sex_df["Sex"] = full.Sex.map(lambda x: 1 if x == "male" else 0)

sex_df.head()
# Create a new variable for every unique value of Embarked

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked.head()
# Create a new variable for every unique value of Embarked

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
# Create dataset

imputed = pd.DataFrame()



# Fill missing values of Age with the average of Age (mean)

imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )



# Fill missing values of Fare with the average of Fare (mean)

imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )



imputed.head()
full_X = pd.concat( [imputed, embarked , pclass, sex_df] , axis=1 )

full_X.head()
from sklearn.cross_validation import train_test_split , StratifiedKFold
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_X[ 0:891 ]

train_valid_y = full.Survived[0:891]

test_X = full_X[ 891: ]

test_y = full



train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
# Score the model

print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
test_Y = model.predict( test_X )

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test["Survived"] = test["Survived"].astype(int)

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )