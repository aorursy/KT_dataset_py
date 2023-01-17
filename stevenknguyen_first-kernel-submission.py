# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(os.path.join('../input', 'train.csv'))

test = pd.read_csv(os.path.join('../input', 'test.csv'))

train.info()
train['Embarked'] = train['Embarked']

# rename values

train['Ports'] = train.Embarked.map({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})

train['Survival'] = train.Survived.map({0: 'Died', 1: 'Survived'})

train['Class'] = train.Pclass.map({1:'First Class', 2:'Second Class', 3:'Third Class'})
fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(10,5))

sns.countplot(x='Ports', data = train, ax = axis1)

sns.countplot(x='Survival', hue = "Ports", data = train, ax= axis2)

embark_perc = train[["Ports", "Survived"]].groupby(['Ports'], as_index = False).mean()

sns.barplot(x='Ports', y = 'Survived', data= embark_perc, order = ['Southampton','Cherbourg','Queenstown'], ax=axis3).set(ylabel='survival rate')
sns.countplot(train['Class'], hue=train['Survival'], order = ['First Class', 'Second Class', 'Third Class'])
train['Fare'].describe()
fare_dead = train['Fare'][train['Survived'] == 0]

fare_survived = train['Fare'][train['Survived']==1]

fare=[fare_dead,fare_survived]



axes=plt.gca()

axes.set_ylim([0,125])

plt.ylabel(['Price of ticket'])

plt.boxplot(fare, labels =['Dead', 'Survived'])
age_dead = train['Age'][train['Survived'] == 0]

age_survived = train['Age'][train['Survived'] ==1]

ages = [age_dead, age_survived]



fig, (axis1, axis2) = plt.subplots(1,2, figsize = (9,5))

axis1.set_title('Ages of dead')

axis2.set_title('Ages of survived')

age_dead.hist(bins=70, ax= axis1).set(ylabel='Count')

age_survived.hist(bins=70, ax= axis2).set(ylabel='Count')
train['Family'] = train["Parch"] + train["SibSp"]

fig, (axis1, axis2) = plt.subplots(1,2,sharex = True, figsize=(9,5))

sns.countplot(x='Family', data=train, order=[1,0], ax= axis1).set(xlabel = 'Family Status')



#ratio of survivors 

family_perc = train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax= axis2).set(xlabel= 'Family Status', ylabel='Survival Rate')

axis1.set_xticklabels(["With Family", "Alone"], rotation=0)
def get_person(passenger):

    age, sex = passenger

    if age< 18:

        if age <2: 

            return 'baby'

        else:

            return 'child'

    else:

        return sex

train['Person'] = train[['Age', 'Sex']].apply(get_person, axis=1)



fig, (axis1, axis2) = plt.subplots(1,2,figsize=(9,5))

sns.countplot(x='Person', data=train, ax=axis1)

person_perc = train[["Person", "Survived"]].groupby(['Person'], as_index = False).mean()

sns.barplot(x = 'Person', y = 'Survived', 

            data=person_perc, ax=axis2, order=['male', 'female','child','baby']).set(ylabel='Survival Rate')
full = train.append( test , ignore_index = True )



sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked = pd.DataFrame()

embarked['Embarked'] = full.Embarked.fillna('S')

embarked[ 'Embarked' ] = embarked[ 'Embarked' ].map( lambda c : c[0] )

embarked = pd.get_dummies( embarked['Embarked'] , prefix = 'Embarked' )

pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

age = pd.get_dummies( full.Age , prefix='Age' )



def fill_fare(passenger):

    pclass, fare = passenger

    if fare is None:

        if pclass == 3:

            return 9

    

imputed = pd.DataFrame()

imputed['Fare'] = full[['Pclass','Fare']].apply(fill_fare, axis=1)

imputed = pd.get_dummies( imputed[ 'Fare' ] , prefix = 'Fare' )

#imputed['Age'] = full.Age.fillna(train.Age.median())



# Fill missing values of Fare with the median

#imputed[ 'Fare' ] = full.Fare.fillna( train.Fare.median() )

# need to fill fare based on the class of the passenger



family = pd.DataFrame()

# introducing a new feature : the size of families 

family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1



# introducing other features based on the family size

#family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

#family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

#family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )



title = pd.DataFrame()

title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



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



cabin = pd.DataFrame()



# replacing missing cabins with U (for Uknown)

cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )



# mapping each Cabin value with the cabin letter

cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )



# dummy encoding ...

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )



def cleanTicket( ticket ):

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len( ticket ) > 0:

        return ticket[0]

    else: 

        return 'XXX'



ticket = pd.DataFrame()



# Extracting dummy variables from tickets:

ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )



def get_person(passenger):

    age, sex = passenger

    if (age < 18):

        if(age <2):

            return 'baby'

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

    

person = pd.DataFrame()

person['Person'] = full[['Age','Sex']].apply(get_person, axis=1)

person = pd.get_dummies( person[ 'Person' ] , prefix = 'Person' )
from sklearn.cross_validation import train_test_split

#full_X = pd.concat( [ title , pclass, person, imputed, family, embarked] , axis=1 )

full_X = pd.concat( [person, title , pclass, family, embarked, imputed, ticket] , axis=1 )



train_valid_X = full_X[ 0:891 ]

train_valid_y = train.Survived

test_X = full_X[ 891: ]

train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)


from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(#criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_depth =10,

                             random_state=1,

                             n_jobs=-1

)

#rfc = RandomForestClassifier(n_estimators=30000, min_samples_leaf=2, class_weight={0:0.745,1:0.255})

rfc.fit( train_X , train_y )
print (rfc.score( train_X , train_y ) , rfc.score( valid_X , valid_y ))
test_Y = rfc.predict( test_X )

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )