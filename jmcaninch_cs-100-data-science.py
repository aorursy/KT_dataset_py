import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:30]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[(train_data.Sex == 'male')]['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

#train_data[['Embarked','Sex','Survived']].groupby(['Embarked','Sex']).mean()

broad_count = 0

spec_count = 0

survivedCount = 0

diedCount = 0

survivedTotal = 0

diedTotal = 0



print('Analyzing pclass...')

for idx, row in train_data.iterrows():

    try:

        if row['Embarked'] == 'S':

            if row['Survived'] == 1:

                survivedCount += 1

                survivedTotal += int(row['Pclass'])

            else:

                diedCount += 1

                diedTotal += int(row['Pclass'])

    except:

        pass

print('Average survivor pclass:',(survivedTotal/survivedCount))

print('Average non-survivor pclass:',(diedTotal/diedCount))

        #broad_count += 1

        #if str(row['Cabin'])[0] not in ['B','D','E']:

        #if row['Age'] > 18:

            #if row['Survived'] == 1:

                #spec_count += 1

#print(str(spec_count)+'/'+str(broad_count)+' = '+str(spec_count/broad_count))
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
spec_count = 0

spec_survived_count = 0

otherSurvivors = 0



for idx, row in train_data.iterrows():

    #if ((row['Age'] < 18 or row['Sex'] == 'female') and row['Pclass']!=3) or ((row['Embarked'] == 'S' and row['Sex'] == 'female') and row['Pclass']!=3):

    if row['Pclass']==3 and (row['Sex']=='male' and row['Parch'] != 0 and row['SibSp'] == 0 and row['Age'] < 40):

        spec_count += 1

        if row['Survived'] == 1:

            spec_survived_count += 1

    elif row['Survived'] == 1:

        otherSurvivors += 1

        

print(str(spec_survived_count)+'/'+str(spec_count))

print(spec_survived_count / spec_count)

print('Other survivors remaining:', otherSurvivors)
s = train_data[(train_data.Embarked == 'S')]

spec_count = 0

spec_survived_count = 0



for idx, row in s.iterrows():

    if row['Sex']=='male' and (row['Parch'] == 0):     

        spec_count += 1

        if row['Survived'] == 1:

            spec_survived_count += 1

            



print(str(spec_survived_count)+'/'+str(spec_count))

print(spec_survived_count / spec_count)
#s = train_data[(train_data.Embarked == 'S')&(train_data.Sex == 'female')&(train_data.Survived == 0)]

#s[['SibSp','Parch']]



partial = train_data[(train_data.Pclass == 3)&(train_data.Sex == 'male')]

partial[['Parch','SibSp','Survived']].groupby(['Parch','SibSp']).mean()

predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if (row['Pclass']!=3):

        if row['Age'] < 18:

            predictions.append(1)

        elif row['Sex'] == 'female':

            predictions.append(1)

        else:

            predictions.append(0)

            

    elif (row['Pclass']==3):

        if str(row['Cabin'])[0] == 'E':

            predictions.append(1)

        elif row['Sex']=='female' and (row['Parch'] == 0 or row['Age']<40):

            predictions.append(1) 

        elif (row['Sex']=='male' and row['Parch'] < 3) and (row['Parch'] != 0 and row['SibSp'] == 0 and row['Age'] < 40):

            predictions.append(1)

        else:

            predictions.append(0)

    else:

        predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)