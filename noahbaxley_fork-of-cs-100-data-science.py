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

train_data[['Sex', 'Survived']].groupby(['Sex']).mean() # get sex the data associated with sex and survived and then separate by sex, then calculate the mean
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean() # analyze the social class
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
count1w = 0

count1m = 0

count2w = 0

count2m = 0

count3w = 0

count3m = 0

countm = 0

countw = 0

# men age

for idx, row in train_data.iterrows():

    if row['Pclass'] == 1:

        if row['Sex'] == 'female':

            count1w += 1

        else:

            count1m += 1

    elif row['Pclass'] == 2:

        if row['Sex'] == 'female':

            count2w += 1

        else:

            count2m += 1

    else:

        if row['Sex'] == 'female':

            count3w += 1

        else:

            count3m += 1

    if row['Sex'] == 'male':

        countm += 1

    else:

        countw += 1



print("Class 1 Men:",count1m/countm)

print("Class 1 Women:",count1w/countw)

print("Class 2 Men:",count2m/countm)

print("Class 2 Women:",count2w/countw)

print("Class 3 Men:",count3m/countm)

print("Class 3 Women:",count3w/countw)

#print(train_data.query({'Sex': 'male'}))
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female':

        if row['Pclass'] == 1 or row['Pclass'] == 2 or row['Age'] < 25.0:

            predictions.append(1)

        else:

            predictions.append(0)

    else:

        if row['Age'] < 18.0 and row['Pclass'] == 1: # Small subsection

            predictions.append(1)

        else:

            predictions.append(0)

            

print(predictions[:10])
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)