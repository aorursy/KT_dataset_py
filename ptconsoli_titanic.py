""" 

Author : Pietro Consoli

Date : 25 January 2017



""" 

%matplotlib inline



import pandas as pd

import numpy as np

import csv as csv

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier



# Data cleanup

# TRAIN DATA

train_df = pd.read_csv('../input/train.csv', header=0)        # Load the training dataset file into a dataframe

print(train_df.describe(include='all'))
#transform categorical data Sex into Gender column

train_df['Gender'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#calculate the median age for females and median age for males and assign it to missing values

if len(train_df[(train_df.Gender==1) & (train_df.Age.isnull()) ]) > 0:

    median_female = train_df[train_df.Gender==1]['Age'].dropna().median()

    train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender==1), 'Age'] = median_female

if len(train_df[(train_df.Gender==0) & (train_df.Age.isnull()) ]) > 0:

    median_male = train_df[train_df.Gender==0]['Age'].dropna().median()

    train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender==0), 'Age'] = median_male



#assign to the missing values of Embarked the most common value and convert to numerical values

train_df.loc[ train_df.Embarked.isnull(),'Embarked'] = train_df.Embarked.dropna().mode().values

train_df['Departure'] = train_df.Embarked.map({'C':0, 'S':1, 'Q':2}).astype(int)





if len(train_df[(train_df.Pclass==1) & (train_df.Fare==0) ]) > 0:

    median_first_fare = train_df[train_df.Pclass==1]['Fare'].dropna().median()

    train_df.loc[ (train_df.Fare==0) & (train_df.Pclass==1), 'Fare'] = median_first_fare

if len(train_df[(train_df.Pclass==2) & (train_df.Fare==0) ]) > 0:

    median_second_fare = train_df[train_df.Pclass==2]['Fare'].dropna().median()

    train_df.loc[ (train_df.Fare==0) & (train_df.Pclass==1), 'Fare'] = median_second_fare

if len(train_df[(train_df.Pclass==3) & (train_df.Fare==0) ]) > 0:

    median_third_fare = train_df[train_df.Pclass==3]['Fare'].dropna().median()

    train_df.loc[ (train_df.Fare==3) & (train_df.Pclass==1), 'Fare'] = median_third_fare
#we extract the deck from the cabin for the existing values

train_df['Deck'] = train_df['Cabin'].dropna().str[0:1]

#assign the most common value for each missing entry based on the class



if len(train_df.Deck.isnull() & train_df.Pclass==1) > 0:

    mode_first_deck = train_df[train_df.Pclass==1]['Deck'].dropna().mode().values

    train_df.loc[ (train_df.Deck.isnull()) & (train_df.Pclass==1), 'Deck'] = mode_first_deck

    train_df.loc[ train_df.Deck=='T', 'Deck'] = mode_first_deck

if len(train_df.Deck.isnull() & train_df.Pclass==2) > 0:

    mode_second_deck = train_df[train_df.Pclass==2]['Deck'].dropna().mode().values

    train_df.loc[ (train_df.Deck.isnull()) & (train_df.Pclass==2), 'Deck'] = mode_second_deck

if len(train_df.Deck.isnull() & (train_df.Pclass==3)) > 0:

    mode_third_deck = train_df[ (train_df.Pclass==3)]['Deck'].dropna().mode().values

    train_df.loc[ (train_df.Deck.isnull()) & (train_df.Pclass==3), 'Deck'] = mode_third_deck



train_df['Deck'] = train_df.Deck.map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}).astype(int)
temp_df= pd.DataFrame({'SibSp':train_df['SibSp'],'Parch':train_df['Parch'],'NRelatives':train_df['SibSp']+train_df['Parch'],\

                      'Survived':train_df['Survived']})

print(temp_df.corr(method='spearman'))
train_df['NRel']=train_df['SibSp']+train_df['Parch']
train_df.drop(['Sex','Name','PassengerId','Ticket','Cabin','Embarked','Pclass','SibSp','Parch'],1, inplace=True)

print(train_df.describe())
fig = pd.tools.plotting.scatter_matrix(train_df, alpha=0.2, figsize=(8, 8), diagonal='')
print(train_df.corr(method='spearman'))
bins=[0,18,32,100]

axes = train_df.hist(column='Age', by=['Survived','Gender'], bins=bins, figsize=(8,10))

titles=['Not Survived/Male','Not Survived/Female','Survived/Male','Survived/Female']

i=0

for axes2 in axes:

    for ax in axes2:

        ax.set_xlim(0,100)

        ax.set_ylim(0,250)

        ax.set_xticks(bins)

        ax.set_yticks(np.arange(0,300,15))

        ax.set_title(titles[i])

        i=i+1

plt.plot()
train_df['Range'] = train_df['Age']

for i,val in enumerate(bins[1:]):

    train_df.loc[ (train_df.Range > bins[i]) & (train_df.Range<= bins[i+1]) & (train_df.Gender==0),'Range']=i

    train_df.loc[ (train_df.Range > bins[i]) & (train_df.Range<= bins[i+1]) & (train_df.Gender==1),'Range']=i+len(bins)-1

    

temp_df= pd.DataFrame({'Range':train_df['Range'],'Age':train_df['Age'],'Survived':train_df['Survived'],'Gender':train_df['Gender']})

print(temp_df.corr(method='spearman'))
test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe



#transform categorical data Sex into Gender column

test_df['Gender'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#calculate the median age for females and median age for males and assign it to missing values

if len(test_df[(test_df.Gender==1) & (test_df.Age.isnull()) ]) > 0:

    median_female = test_df[test_df.Gender==1]['Age'].dropna().median()

    test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender==1), 'Age'] = median_female

if len(test_df[(test_df.Gender==0) & (test_df.Age.isnull()) ]) > 0:

    median_male = test_df[train_df.Gender==0]['Age'].dropna().median()

    test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender==0), 'Age'] = median_male



#assign to the missing values of Embarked the most common value and convert to numerical values

test_df.loc[ test_df.Embarked.isnull(),'Embarked'] = test_df.Embarked.dropna().mode().values

test_df['Departure'] = test_df.Embarked.map({'C':0, 'S':1, 'Q':2}).astype(int)



if len(test_df[(test_df.Pclass==1) & ((test_df.Fare==0) | (test_df.Fare.isnull())) ]) > 0:

    median_first_fare = test_df[test_df.Pclass==1]['Fare'].dropna().median()

    test_df.loc[ ((test_df.Fare.isnull()) | (test_df.Fare==0)) & (test_df.Pclass==1), 'Fare'] = median_first_fare

if len(test_df[(test_df.Pclass==2) & ((test_df.Fare==0) | (test_df.Fare.isnull())) ]) > 0:

    median_second_fare = test_df[test_df.Pclass==2]['Fare'].dropna().median()

    test_df.loc[ ((test_df.Fare.isnull()) | (test_df.Fare==0)) & (test_df.Pclass==2), 'Fare'] = median_second_fare

if len(test_df[(test_df.Pclass==3) & ((test_df.Fare==0) | (test_df.Fare.isnull())) ]) > 0:

    median_third_fare = test_df[test_df.Pclass==3]['Fare'].dropna().median()

    test_df.loc[ ((test_df.Fare.isnull()) | (test_df.Fare==0)) & (test_df.Pclass==3), 'Fare'] = median_third_fare



#we extract the deck from the cabin for the existing values

test_df['Deck'] = test_df['Cabin'].dropna().str[:1]



#assign the most common value for each missing entry based on the class - later you can try to predict using fare and class

#there is an entry with deck = T, we interpret as an error. replace with the mode for its passenger class 



if len(test_df.Deck.isnull() & test_df.Pclass==1) > 0:

    mode_first_deck = test_df[test_df.Pclass==1]['Deck'].dropna().mode().values

    test_df.loc[ (test_df.Deck.isnull()) & (test_df.Pclass==1), 'Deck'] = mode_first_deck

if len(test_df.Deck.isnull() & test_df.Pclass==2) > 0:

    mode_second_deck = test_df[test_df.Pclass==2]['Deck'].dropna().mode().values

    test_df.loc[ (test_df.Deck.isnull()) & (test_df.Pclass==2), 'Deck'] = mode_second_deck

if len(test_df.Deck.isnull() & (test_df.Pclass==3)) > 0:

    mode_third_deck = test_df[ (test_df.Pclass==3)]['Deck'].dropna().mode().values

    test_df.loc[ (test_df.Deck.isnull()) & (test_df.Pclass==3), 'Deck'] = mode_third_deck



test_df['Deck'] = test_df.Deck.map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}).astype(int)

    

test_df['NRel']=test_df['SibSp']+test_df['Parch']



test_df['Range'] = train_df['Age']

for i,val in enumerate(bins[1:]):

    test_df.loc[ (test_df.Range > bins[i]) & (test_df.Range<= bins[i+1]) & (test_df.Gender==0),'Range']=i

    test_df.loc[ (test_df.Range > bins[i]) & (test_df.Range<= bins[i+1]) & (test_df.Gender==1),'Range']=i+len(bins)-1





# Collect the test data's PassengerIds before dropping it

ids = test_df['PassengerId'].values



test_df.drop(['Sex','Name','PassengerId','Ticket','Cabin','Embarked','Pclass','SibSp','Parch'],1, inplace=True)


train_data = train_df.values

test_data = test_df.values



print('Training...')

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0:,1:], train_data[0::,0] )



print('Predicting...')

output = forest.predict(test_data).astype(int)

print(forest.score(train_data[0:,1:], train_data[0::,0]))

predictions_file = open("myfirstforest.csv", "w", newline="")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(['PassengerId','Survived'])

open_file_object.writerows(zip(ids, output))

predictions_file.close()

print('Done.')
