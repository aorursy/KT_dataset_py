import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
women = train_data[train_data['Embarked'] == 'C']['Survived']

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

    if row['Sex'] == 'male':

        if row['Pclass'] == 1:  #best for class male

             if row['Age'] <= 14:

        #if row['Age'] <= 12:   #best for young people male

        #if row['Fare'] >= 75:  #best for fare male

        #if row['Parch'] >= 1:  #best for Parch of >=1 of 31% male

                

                women_count += 1

                if row['Survived'] == 1:

                    women_survived_count += 1

                    

#                 if row['Parch'] > 1:

#                     women_count += 1

#                     if row['Survived'] == 1:

#                         women_survived_count += 1

print(women_count)

women_survived_count / women_count
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        if row['Pclass'] == 3:

            women_count += 1

            if row['Age'] > 39:



                women_count += 1

                if row['Survived'] == 1:

                    women_survived_count += 1

                    

#                 if row['Fare'] > 12:

#                     women_count += 1

#                     if row['Survived'] == 1:

#                         women_survived_count += 1

print(women_count)

women_survived_count / women_count
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'male':

        if row['Age'] <= 5:

            women_count += 1

            if row['Survived'] == 1:

                women_survived_count += 1



            

print(women_count)

women_survived_count / women_count
predictions = []

for idx, row in test_data.iterrows():

    if row['Age'] == 1:

        predictions.append(1)

    elif row['Age'] <= 10 and row['Pclass'] == 2: # children survives

        predictions.append(1)

    else:

        if row['Sex'] == 'female':

            if row['Pclass'] == 3: # Pclass of 3 for females have 50% survival chance

                if row['Age'] >= 39:

                    predictions.append(0)

                elif row['Age'] > 27:

                    if row['Fare'] > 15:

                        predictions.append(1)

                    else:

                        predictions.append(0)

                elif row['Age'] > 17:

                    predictions.append(1) # all fare of ages 19 to 27 has above 50% survival chance

                elif row['Age'] > 11:

                    if row['Fare'] > 12:

                        predictions.append(0)

                    else:

                        predictions.append(1)

                elif row['Age'] > 5:

                    if row['Fare'] > 12:

                        predictions.append(0)

                    else:

                        predictions.append(1)

                else:

                    if row['Fare'] > 12:

                        predictions.append(0)

                    else:

                        predictions.append(1)

            else: # first and second class for females basically all live

                predictions.append(1)

        else:

            if row['Pclass'] == 1:

                if row['Age'] >= 36:

                    predictions.append(0)

                elif row['Age'] <= 14:

                    predictions.append(1)

                else:

                    predictions.append(0)

#                 elif row['Age'] > 30:

#                     predictions.append(1)

#                 elif row['Age'] > 27:

#                     if row['Fare'] < 40:

#                         predictions.append(1)

#                     else:

#                         predictions.append(0)

#                 elif row['Age'] > 19:

#                     if row['Fare'] < 100:

#                         predictions.append(1)

#                     else:

#                         predictions.append(0)

#                 elif row['Age'] > 11:

#                     predictions.append(0)

#                 else:

#                     predictions.append(1)

            else:

                predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)