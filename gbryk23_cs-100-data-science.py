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
train_data[['Fare', 'Survived']].groupby(['Fare']).mean()
train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean()
train_data[['Parch', 'Survived']].groupby(['Parch']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
count = 0

survived_count = 0

best = 0

besti = 0

for i in range(1,60):

    count = 0

    survived_count = 0

    for idx, row in train_data.iterrows():

        if(row['Sex'] == 'female' and row['Pclass']==3 and row['Age']>i):

            count += 1

            if row['Survived'] == 0:

                survived_count += 1

    if(count != 0):

        result = survived_count / count

    else:

        result = 0

    if(result>best):

        best = result

        besti = i

print(best)

print(besti)
count = 0

survived_count = 0

for idx, row in train_data.iterrows():

    if(row['Sex'] == 'female' and row['Pclass']==3 and row['Age']>38):

        count += 1

        if row['Survived'] == 0:

            survived_count += 1

if(count != 0):

    print(survived_count / count)
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if (row['Sex'] == 'female'):

        if(row['Pclass']==3 and row['Age']>38):

            predictions.append(0)

        else:

            predictions.append(1)



    else:  

        if(row['Pclass']<=2 and row['Age'] <= 15):

            predictions.append(1)

        elif(row['SibSp']<=1 and row['Age']<=14):

            predictions.append(1)

        elif(row['Fare']>100 and row['Age']<=16):

            predictions.append(1)

        else:

            predictions.append(0)

assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)