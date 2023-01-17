import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Parch'] == 0 and row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count

count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    count += 1

    if row['Sex'] == 'female':

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / count



predictions = []

for idx, row in test_data.iterrows():

    if row['Sex'] == 'female': #female

        if row['Pclass'] ==3:

            if row['Parch'] >= 3 or row['SibSp'] >= 3 or row['Age'] <= 6:

                predictions.append(0)

            elif row['Parch'] == 0 and row['SibSp'] ==1 and row['Embarked'] != 'Q':

                predictions.append(0)

            elif row['Parch'] == 1 and row['SibSp'] ==3 and row['Embarked'] == 'S':

                predictions.append(0)

            else:

                predictions.append(1)

        else:

            predictions.append(1)

    else: #male

        if row['Age'] <=8 and row['Pclass'] ==2:

            predictions.append(1)

        elif row['Age'] <= 12 and row['Pclass'] ==1:

            predictions.append(1)

        else:

            predictions.append(0)

# my approach: from studying the data, most survivors are female and most females survive, so it is safe to start by marking females with 1s and males with 0s

# after studying the data from sorting, plotting and graphing it in multiple ways, I found that:

# females in classes 1 and 2 are dominantly surviving, and half of 3rd class women are dying. A majority of them share the one of the following characteristics (Parch >= 3, SibSp >=3, Age <= 6)

# for males, most don't survive. Fore more accuracy, young males have better survival change, but that can't be applied as a general rule (will decrease accuracy)

# after categorizing males in classes, we find class 1 males with ages <=17 all survived, and class 2 has males with ages <=8 all survived as well. Class 3 males don't have an age majority of survival.
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)