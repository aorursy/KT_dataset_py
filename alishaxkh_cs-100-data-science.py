import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
total_survived = sum(train_data[train_data['Sex'] == 'female']['Survived'])

print(total_survived)

print(len(train_data[train_data['Sex'] == 'female']))
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[(train_data['Parch'] > 0 )| (train_data['SibSp'] > 0)][['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).agg(['sum','count','mean'])
train_data[train_data['Fare'] > 30][['Sex','Pclass','Survived']].groupby(['Sex','Pclass']).agg(['sum','count','mean'])
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[(train_data.Sex == 'female') & (train_data.Pclass == 3)].describe()
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
#predictions = []

#for idx, row in test_data.iterrows():

    # make your changes in this cell!

 #   if row['Sex'] == 'female':

  #      predictions.append(1)

   #     #predictions append 1 means they survive

    #elif row['Sex'] == 'male':

     #   if row['Age'] <= 10 :

      #      predictions.append(1)

       # else:

        #    predictions.append(0)

    #if row['Sex']=='male' and row['Age']<=10:

    #    predictions.append(1)

        #predictions append 0 means they die

    #if row['Sex']=='male'and row['Age']>10:

    #    predictions.append(0)

    
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female':

        predictions.append(1)

        #predictions append 1 means they survive

    elif row['Sex'] == 'male':

        if row['Age'] <= 10 or ((row['Parch'] >0 or row['SibSp']>0) & (row ['Pclass']<3)):

            predictions.append(1)

        else:

            predictions.append(0)

      

    #if row['Sex']=='male' and row['Age']<=10:

    #    predictions.append(1)

        #predictions append 0 means they die

    #if row['Sex']=='male'and row['Age']>10:

    #    predictions.append(0)

    
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)