import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:22]
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
fc1 = 0

fc2 = 0

fc3 = 0

mc1 = 0

mc2 = 0

mc3 = 0

fc1s = 0

fc2s = 0

fc3s = 0

mc1s = 0

mc2s = 0

mc3s = 0



countSurv = 0

countDied = 0

totalAgeSurv = 0

totalAgeDied = 0



AgeSurv = []

AgeDied = []



for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        if row['Pclass'] == 1:

            fc1 += 1

            if row['Survived'] == 1:

                fc1s += 1

        if row['Pclass'] == 2:

            fc2 += 1

            if row['Survived'] == 1:

                fc2s += 1

        if row['Pclass'] == 3:

            fc3 += 1

            if row['Survived'] == 1:

                fc3s += 1

                try:

                    totalAgeSurv += int(row['Age'])

                    AgeSurv.append(row['Age'])

                    countSurv += 1

                except:

                    pass

            elif row['Survived'] == 0:

                try:

                    totalAgeDied += int(row['Age'])

                    AgeDied.append(row['Age'])

                    countDied += 1

                except:

                    pass

    elif row['Sex'] == 'male':

        if row['Pclass'] == 1:

            mc1 += 1

            if row['Survived'] == 1:

                mc1s += 1

        if row['Pclass'] == 2:

            mc2 += 1

            if row['Survived'] == 1:

                mc2s += 1

        if row['Pclass'] == 3:

            mc3 += 1

            if row['Survived'] == 1:

                mc3s += 1



                

print(max(AgeSurv))

print('')

print(max(AgeDied))

                

'''

print(totalAgeSurv/countSurv)

print(totalAgeDied/countDied)



print(fc1)

print(fc1s/fc1)

print(fc2)

print(fc2s/fc2)

print(fc3)

print(fc3s/fc3)

print(mc1)

print(mc1s/mc1)

print(mc2)

print(mc2s/mc2)

print(mc3)

print(mc3s/mc3)

'''
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
maleAgeCount1 = 0

femaleAgeCount1 = 0

maleAgeCount0 = 0

femaleAgeCount0 = 0

maleCount1 = 0

femaleCount1 = 0

maleCount0 = 0

femaleCount0 = 0



AgeSurv = []

AgeDied = []



for idx, row in train_data.iterrows():

    try:

        if row['Survived'] == 1:

            if row['Sex'] == 'female':

                femaleCount1 += 1

                AgeSurv.append(row['Age'])

                femaleAgeCount1 += int(row['Age'])

            else:

                maleCount1 += 1

                maleAgeCount1 += int(row['Age'])

        else:

            if row['Sex'] == 'female':

                femaleCount0 += 1

                AgeDied.append(row['Age'])

                femaleAgeCount0 += int(row['Age'])

            else:

                maleCount0 += 1

                maleAgeCount0 += int(row['Age'])

    except:

        pass

print(femaleAgeCount1/femaleCount1)

print(maleAgeCount1/maleCount1)

print(femaleAgeCount0/femaleCount0)

print(maleAgeCount0/maleCount0)



print(AgeSurv)

print('')

print(AgeDied)

print('')

print(max(AgeSurv))

print(max(AgeDied))
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female' and row['Age'] > 5:

        predictions.append(1)

    else:

        predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)