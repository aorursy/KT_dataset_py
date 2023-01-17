import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:30]
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
train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean()
train_data[['Age', 'Survived']].groupby(['Age']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

women_older = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

            if row['Age'] > 18:

                women_older += 1

print(women_older/women_survived_count)

print(women_survived_count/women_count)
women_class3 = 0

women_older = 0

women_survived = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        if row['Pclass'] == 3:

            women_class3 += 1

            if row['Survived'] == 1:

                women_survived += 1

                if row['Age'] <= 1:

                    women_older += 1

print(women_class3)

print(women_survived)

print(women_older)
class_count = 0

class_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Embarked'] == 'C':

        class_count += 1

        if row['Age'] > 18:

            class_survived_count += 1

class_survived_count/class_count
fare_count = 0

survived_count = 0

for idx, row in train_data.iterrows():

     if row['Survived'] == 1: 

        survived_count += 1

        if row['Fare'] < 40:

            fare_count += 1

fare_count/survived_count
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female':

        if row['Pclass'] < 3:

            predictions.append(1)

        elif row['Pclass'] == 3 and row['Age'] <= 1:

            predictions.append(1)

        else:

            predictions.append(0)

    elif row['Age'] < 18 and row['Embarked'] == 'C':

            predictions.append(1)

    else:

            predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)