import pandas as pd

import numpy as np



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]

PassengerId = test_df['PassengerId']



for dataset in combine:

    #Pclass

    dataset['upperClass'] = np.where(dataset['Pclass']==1,1,0)

    dataset['middleClass'] = np.where(dataset['Pclass']==2,1,0)

    dataset['lowerClass'] = np.where(dataset['Pclass']==3,1,0)

    #Title

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Mr'] = np.where(dataset['Title']=='Mr',1,0)

    dataset['Miss'] = np.where(dataset['Title']=='Miss',1,0)

    dataset['Mrs'] = np.where(dataset['Title']=='Mrs',1,0)

    dataset['Master'] = np.where(dataset['Title']=='Master',1,0)

    dataset['rareTitle'] = np.where(dataset['Title']=='Rare',1,0)

    #Gender

    dataset['female'] = np.where(dataset['Sex']=='female',1,0)

    dataset['male'] = np.where(dataset['Sex']=='male',1,0)

    #Cabin

    dataset['CabinChar'] = dataset['Cabin'].str[:1]

    dataset['A'] = np.where(dataset['CabinChar']=='A',1,0)

    dataset['B'] = np.where(dataset['CabinChar']=='B',1,0)

    dataset['C'] = np.where(dataset['CabinChar']=='C',1,0)

    dataset['D'] = np.where(dataset['CabinChar']=='D',1,0)

    dataset['E'] = np.where(dataset['CabinChar']=='E',1,0)

    dataset['noCabin'] = np.where(dataset['Cabin'].isnull(),1,0)

    #Embarked

    dataset['Cherbourg'] = np.where(dataset['Embarked']=='C',1,0)

    dataset['Queenstown'] = np.where(dataset['Embarked']=='Q',1,0)

    dataset['Southampton'] = np.where(dataset['Embarked']=='S',1,0)

    #No Family

    dataset['noFamily'] = np.where(dataset['SibSp'] + dataset['Parch']==0,1,0)

    dataset['familySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df['Survived'] = train_df['Survived'].replace(0,-1)

    

# Average age on Title

all_df = pd.concat([train_df, test_df])

ageGroup = all_df.groupby('Title')['Age'].mean()

# Ticket information

ticketSize = all_df.groupby('Ticket')['PassengerId'].count()



for dataset in combine:

    # Ticket Size, Ticket Survived %

    dataset['ticketSize'] = dataset['Ticket'].map(ticketSize)

    dataset['noTicketPartner'] = np.where(dataset['ticketSize']==1,1,0)

    # Null

    dataset['noAge'] = np.where(dataset['Age'].isnull(),1,0)

    dataset['Age'] = dataset['Age'].fillna(dataset['Title'].map(ageGroup))

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    # Log

    dataset['Age'] = np.where(dataset['Age'] < 1, 1, dataset['Age'])

    dataset['Age'] = np.log(dataset['Age'])

    dataset['Fare'] = np.where(dataset['Fare'] < 1, 1, dataset['Fare'])

    dataset['Fare'] = np.log(dataset['Fare'])

    

"""

Need to calculate the reference surived rate group by ticket

Calculation need to exclude himself, otherwise the field is already have the survive data

"""

ticketInTrain = train_df.groupby('Ticket')['PassengerId'].count()

ticketSurvived = train_df.groupby('Ticket')['Survived'].sum()/ train_df.groupby('Ticket')['PassengerId'].count()



train_df['noTicketRef'] = np.where(train_df['Ticket'].map(ticketInTrain)==1,1,0)

test_df['noTicketRef'] = np.where(test_df['Ticket'].map(ticketInTrain)>0,0,1)

train_df['ticketRef'] = np.where(train_df['noTicketRef']==1,0

        ,(train_df['Ticket'].map(ticketSurvived) * train_df['Ticket'].map(ticketInTrain) - train_df['Survived'])/(train_df['Ticket'].map(ticketInTrain) - 1))

test_df['ticketRef'] = np.where(test_df['noTicketRef']==1,0,test_df['Ticket'].map(ticketSurvived))



train_result = train_df['Survived']

train_df.to_csv('all.csv', index=False)

#Drop

train_df = train_df.drop(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Title', 'Ticket', 'Embarked', 'CabinChar', 'Cabin', 'noTicketRef'], axis=1)

test_df = test_df.drop(['PassengerId', 'Pclass', 'Name', 'Title', 'Sex', 'Ticket', 'Embarked', 'CabinChar', 'Cabin', 'noTicketRef'], axis=1)



train_df.head()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# create some data

train_df = train_df.values

train_result = train_result.values

test_df = test_df.values



x = train_df

y = train_result

z = test_df



# build a neural network from the 1st layer to the last layer

model = Sequential()

model.add(Dense(units=58, input_dim=29, activation='selu'))

model.add(Dropout(0.5))

model.add(Dense(units=29, activation='selu')) 

model.add(Dropout(0.5))

model.add(Dense(units=1, activation='tanh'))



# choose loss function and optimizing method

model.compile(loss='mse', optimizer='sgd')



# training

print('Training -----------')

for step in range(100001):

    cost = model.train_on_batch(x, y)

    if step % 10000 == 0:

        print('step', step, 'train cost:', cost)



# predict

test_predict = model.predict(z)
# Generate Submission File

test_predict = np.where(test_predict>0,1,0)

NNSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': test_predict.ravel() })

NNSubmission.to_csv("NNSubmission.csv", index=False)