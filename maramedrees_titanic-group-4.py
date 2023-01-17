import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



import sys

!{sys.executable} -m pip install pandas_profiling

from pandas_profiling import ProfileReport
def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.describe()
train.info()
train.shape
profile_report = ProfileReport(train, title='Profile Report', html={'style':{'full_width':True}})
profile_report.to_notebook_iframe()
plot_distribution( train , var = 'Age' , target = 'Survived' , row = 'Sex' )
plot_distribution( train , var = 'Fare' , target = 'Survived' )
train.isnull().sum().sum()
test.isnull().sum().sum()
# Percentile of missing values

missing_values_count = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)

plt.figure(figsize=(15,10))

plt.xlabel('Features', fontsize=15)

plt.ylabel('Missing Values', fontsize=15)

plt.title('Percentile of Missing Values', fontsize=15)

sns.barplot(missing_values_count[:10].index.values, missing_values_count[:10], color ='blue');
# Count and percentile of missing data

train_missing = train.isnull().sum()

train_total = train.isnull().sum().sort_values(ascending=False)

train_percent = (train.isnull().sum() * 100 /train.isnull().count()).sort_values(ascending=False)

train_missing_data = pd.concat([train_total, train_percent], axis=1, keys=['Total', 'Percent'])

train_missing_data.head(3)
test_missing = test.isnull().sum()

test_total = test.isnull().sum().sort_values(ascending=False)

test_percent = (test.isnull().sum() * 100 /test.isnull().count()).sort_values(ascending=False)

test_missing_data = pd.concat([test_total, test_percent], axis=1, keys=['Total', 'Percent'])

test_missing_data.head(4)
train.columns
test.columns
sns.boxplot(x=train['Age'])
sns.boxplot(x=train['SibSp'])
sns.boxplot(x=train['Parch'])
sns.boxplot(x=train['Fare'])
# Replacing missing 'Age' with median

train['Age'].fillna(train['Age'].median(), inplace = True)

test['Age'].fillna(train['Age'].median(), inplace = True)
# Replacing missing 'Fare' data with mean

fare_mean = np.round(train['Fare'].mean())



train['Fare'] = train['Fare'].fillna(fare_mean)

test['Fare'] = test['Fare'].fillna(fare_mean)
# Combining Datasets

y = train['Survived']



full_set = train.append( test , ignore_index = True)

full_set.drop(columns=['Survived'], inplace=True)



print ('Datasets:' , 'full_set:' , full_set.shape)
# Replacing missing 'Embarked' data with mode

full_set['Embarked'] = full_set['Embarked'].fillna(full_set['Embarked'].mode()[0])
title = pd.DataFrame()



# Extracting the 'Title' from each name

title[ 'Title' ] = full_set[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



# Droping the 'Name' from the original dataset then adding it again in its new processed form

full_set.drop(columns=['Name'], inplace=True)



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



# Mapping each title

title[ 'Title' ] = title.Title.map( Title_Dictionary )



title.head()
cabin = pd.DataFrame()



# Replacing missing cabins with U (for unknown) to enhance the model's performance and maintain data size

cabin[ 'Cabin' ] = full_set.Cabin.fillna( 'U' )



# Dropping the 'Cabin' from the original dataset then adding it again in its new processed form

full_set.drop(columns=['Cabin'], inplace=True)



# Mapping each 'Cabin' value with the 'Cabin' letter

cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )



cabin.head()
# Adding a function that extracts each prefix of the ticket, and returns 'XX' if no prefix (i.e the ticket is a digit)

def cleanTicket( ticket ):

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len( ticket ) > 0:

        return ticket[0]

    else: 

        return 'XX'



ticket = pd.DataFrame()





ticket[ 'Ticket' ] = full_set[ 'Ticket' ].map( cleanTicket )



# Dropping the 'Ticket' feature from the original dataset then adding it again in its new processed form

full_set.drop(columns=['Ticket'], inplace=True)



ticket.head()
family = pd.DataFrame()



# Creating a new feature called 'FamilySize' that includes the size of families including passengers

family[ 'FamilySize' ] = full_set[ 'Parch' ] + full_set[ 'SibSp' ] + 1

# Dropping the 'Parch' and 'SibSp' features from the original dataset then adding them again in their new processed form

full_set.drop(columns=['Parch','SibSp'], inplace=True)



# Creating other features based on the 'FamilySize' feature

family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )

family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )

family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )



family.head()
# Assembling the 'full_X' dataset by adding cleaned features



full_X = pd.concat( [full_set,cabin, title, ticket, family ] , axis=1 )

full_X['Sex'] = full_X['Sex'].apply(lambda x: 1 if x=='female' else 0)

full_X.rename(columns = {'Sex':'isFemale'}, inplace=True)

full_X.head()
# Binning the 'Age' and 'Fare' features to improve the model's performance

full_X['FareBin'] = pd.qcut(full_X['Fare'], 6)

full_X['AgeBin'] = pd.cut(full_X['Age'].astype(int), 7)



full_X.drop(columns=['Age','Fare'],inplace=True)
print(full_X['AgeBin'])
# Converting categorical variables into dummy variables to maintain the size of the data and increase the model's performance

test_id = test['PassengerId']

y = train['Survived'].reset_index(drop=True)



full_X_dummy = pd.get_dummies(full_X, drop_first=True)
# Splitting the data into train and test

train_cleaned = full_X_dummy.iloc[:len(y), :] # Train

test_cleaned = full_X_dummy.iloc[len(y):, :]  # Test
train_cleaned.shape
test_cleaned.shape
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
# Scaling the features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_cleaned)

train_scaled = scaler.transform(train_cleaned)

test_scaled = scaler.transform(test_cleaned)
X_train, X_test, y_train, y_test = train_test_split(train_scaled, y,test_size = .3, random_state = 0)
# Fit Decision Trees

dt = DecisionTreeClassifier()

score_CV = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy',n_jobs=-1)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Decision Tree", "Train", score_CV.mean().round(3), score_CV.std().round(3)))
# Fit Random Forest

rf = RandomForestClassifier()

score_CV = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Random Forest", "Train", score_CV.mean().round(3), score_CV.std().round(3)))
dt = ExtraTreesClassifier()

score_CV = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy',  n_jobs=-1)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Extra Trees", "Train", score_CV.mean().round(3), score_CV.std().round(3)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

score_CV = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("KNeighbors", "Train", score_CV.mean().round(3), score_CV.std().round(3)))
# Fit Bagging + Decision Trees

dt_bag = BaggingClassifier(knn)

score_CV = cross_val_score(dt_bag, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

print("{} {} Score:\t{:0.3} ± {:0.3}".format("Bagging", "Train", score_CV.mean().round(3), score_CV.std().round(3)))
# Using the best scoring model

dt_bag.fit(X_train, y_train)

dt_bag.score(X_test, y_test)
# Using the best scoring model

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
pred_y = knn.predict(test_scaled)
pred = pd.DataFrame()

pred['PassengerId'] = test_id

pred['Survived'] = pred_y

pred.to_csv('22nd__model.csv',index=False)