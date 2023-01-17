import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
C1mean = train_data[train_data['Pclass']==1].Fare.mean()

cls1 = train_data[train_data['Pclass']==1][train_data.Fare > C1mean]['Survived']

rate_fare = sum(cls1)/len(cls1)

print('% of C1 people who paid less than C1 mean and survived:', rate_fare)
C2mean = train_data[train_data['Pclass']==2].Fare.mean()

cls1 = train_data[train_data['Pclass']==2][train_data.Fare > C2mean]['Survived']

rate_fare = sum(cls1)/len(cls1)

print('% of C2 people who paid less than C2 mean and survived:', rate_fare)
C3mean = train_data[train_data['Pclass']==3].Fare.mean()

cls1 = train_data[train_data['Pclass']==3][train_data.Fare > C3mean]['Survived']

rate_fare = sum(cls1)/len(cls1)

print('% of C3 people who paid less than C3 mean and survived:', rate_fare)
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
fare = train_data[train_data.Fare < 32.204208]['Survived']

rate_fare = sum(fare)/len(fare)

print('% of people who paid less than 32 and survived:', rate_fare)
train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data[train_data.Sex == 'male'].corr().Survived.Pclass
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
train_data[train_data.Sex == 'male'].corr().Survived.Pclass
train_data[train_data.Pclass == 1].Fare.mean()
predictions = []

for idx, row in test_data.iterrows():

    if row['Sex'] == 'female' or (row['Pclass'] == 1 and row['Fare'] > 85):

        predictions.append(1)

    else:

        predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)