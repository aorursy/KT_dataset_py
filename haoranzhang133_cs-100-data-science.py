import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:20]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
age = train_data[train_data['Age'] >= 60 ]['Survived']

rate_age = sum(age)/len(age)

print('% of age larger than 40 who survived:', rate_age)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)



# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train_data[['Fare', 'Survived']].groupby(['Fare']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'male':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
st = '123'

type(int(st))
Q = 0

qppl = 0

Q_sur = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'male':

        if row['Parch']== 2:

            Q +=1

            if row['Survived'] == 1:

                Q_sur +=1

Q_sur/Q

test_data
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Cabin'] == 'B96 B98':

        predictions.append(1)

    elif row['Sex'] == 'female':

        if row['Age'] <=11 and row['Age'] >=6:

            predictions.append(0)

        elif row['Cabin'] == 'E101' or row['Cabin'] == 'F33':

            predictions.append(1)

        elif row['Parch'] >=4 or row['SibSp'] >=3 :

            predictions.append(0)

        elif (row['Fare'] >=8 and row['Fare'] <=10) or (row['Fare'] >=14 and row['Fare'] <=15):

            predictions.append(0)

        elif (row['Pclass'] == 3 and row['Embarked']=='S') or (row['Pclass'] == 3 and row['Embarked'] == 'Q'):

            

            predictions.append(0)

        else:

            predictions.append(1)

            

    else:

        

        if row['Age'] <=15:

            if row['Parch'] == '2'or row['Embarked'] == 'C':

                predictions.append(1)

            else:

                predictions.append(0)

            

        elif row['Fare'] >=273:

            predictions.append(1)

        elif row['Age'] >=75:

            predictions.append(1)

        else:

            predictions.append(0)

        

        

    

        

   

   

        

   

    

    

        

        
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)