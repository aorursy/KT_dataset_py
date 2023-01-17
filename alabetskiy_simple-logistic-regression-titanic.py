
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv', index_col="PassengerId")
test_df = pd.read_csv('../input/test.csv', index_col="PassengerId")
#Lets create the Column Survived in the Test data set
test_df['Survived'] = -777
#Lets combain two Test data set and Train data set all together 
dataset = pd.concat((train_df, test_df),axis=0)
dataset.info()
dataset[dataset.Embarked.isnull()]
# how many people embarked at different points
dataset.Embarked.value_counts()
dataset.groupby(['Pclass', 'Embarked']).Fare.median()
# replace missing values with 'C'
dataset.Embarked.fillna('C', inplace=True)
dataset[dataset.Fare.isnull()]
dataset.groupby(['Pclass', 'Embarked']).Fare.median()
dataset.Fare.fillna(8.05, inplace=True)
dataset[dataset.Age.isnull()]
# Function to extract the title from the name 
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title
# lets use map function to assosiate each Name with Title
dataset.Name.map(GetTitle)
dataset.Name.map(lambda x : GetTitle(x)).unique()
# Function to extract the title from the name 
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

# create Title feature
dataset['Title'] =  dataset.Name.map(lambda x : GetTitle(x))

# Box plot of Age with title
dataset[dataset.Age.notnull()].boxplot('Age','Title');

# replace missing values
title_age_median = dataset.groupby('Title').Age.transform('median')
dataset.Age.fillna(title_age_median , inplace=True)

# histogram for fare 
dataset.Fare.plot(kind='hist',title='histogram for fare', bins=20)
# binning. We are splitting some different ranges of values into 4 bins
pd.qcut(dataset.Fare,4)
pd.qcut(dataset.Fare, 4, labels=['very_low','low','high','very_high']) #discretization
# create fare feature bin
dataset['Fare_Bin']=pd.qcut(dataset.Fare, 4, labels=['very_low','low','high','very_high'])
#Age State based on Age
dataset['AgeState']=np.where(dataset['Age'] >= 18, 'Adult', 'Child')
# AgeState counts
dataset['AgeState'].value_counts()

# cross tab
pd.crosstab(dataset[dataset.Survived != -777].Survived, dataset[dataset.Survived !=-777].AgeState)
#family size
# Family: Adding parents with siblings
dataset['FamilySize'] = dataset.Parch + dataset.SibSp + 1 # 1 for self
# lets create crosstable to see a family size impact on survival rate
pd.crosstab(dataset[dataset.Survived != -777].Survived, dataset[dataset.Survived != -777].FamilySize)
#Feature IsMother
# a lady aged more than 18 which has parch>0 and is married (not miss). 1 and 0 in the end of where function means
# if condition is true -> assign 1 else assign 0
dataset['IsMother']=np.where(((dataset.Sex=='female') & (dataset.Parch>0) & (dataset.Age>18) & (dataset.Title !='Miss')),1,0)
# crosstab with IsMother
pd.crosstab(dataset[dataset.Survived !=-777].Survived, dataset[dataset.Survived !=-777].IsMother)
dataset.loc[dataset.Cabin =='T','Cabin'] = np.NaN
# extract first character of Cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
dataset['Deck'] = dataset['Cabin'].map(lambda x : get_deck(x))
#check counts
dataset.Deck.value_counts()
# use crosstab to look into survived feature cabin wise
pd.crosstab(dataset[dataset.Survived !=-777].Survived,dataset[dataset.Survived != -777].Deck)
# sex
dataset['IsMale'] = np.where(dataset.Sex == 'male', 1, 0)
# columns Deck, Pclass, Title, AgeState
dataset = pd.get_dummies(dataset,columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
dataset.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1,inplace=True)
columns = [column for column in dataset.columns if column != 'Survived']
columns = ['Survived'] + columns
dataset = dataset[columns]
dataset.info()
X_train = dataset.loc[dataset['Survived'] != -777, 'Age':'AgeState_Child'].values
y_train = dataset.loc[dataset['Survived'] != -777, 'Survived'].values
X_test = dataset.loc[dataset['Survived'] == -777, 'Age':'AgeState_Child']
# Fitting logistic regression in to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
test_df.head()
data_to_submit = pd.DataFrame({
    'PassengerId':X_test.index,
    'Survived':y_pred
})
data_to_submit.to_csv('csv_to_submit.csv', index = False)