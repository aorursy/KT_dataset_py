import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men_data = train_data



indexNames1 = men_data[ men_data['Sex'] != "male" ].index

men_data = men_data.drop(indexNames1 , inplace=False)



women_p_data = women_ages

indexNames2 = women_p_data[ women_p_data['Pclass'] != 3 ].index

women_p_data = women_p_data.drop(indexNames2 , inplace=False)
women_p_data[:100]
women_ages = train_data



indexNames = women_ages[ women_ages['Sex'] != "female" ].index

women_ages = women_ages.drop(indexNames , inplace=False)



women_array = women_ages[train_data['Survived'] == 1]['Age']

women_array1 = women_ages[train_data['Survived'] == 0]['Age']



import numpy as np; np.random.seed(13)

import matplotlib.pyplot as plt





plt.hist(women_array, bins=len(women_array), ec="k")



plt.show()
plt.hist(women_array1, bins=len(women_array1), ec="k")



plt.show()
men_ages = men_data[train_data['Survived'] == 1]['Age']

men_ages1 = men_data[train_data['Survived'] == 0]['Age']



import numpy as np; np.random.seed(13)

import matplotlib.pyplot as plt





plt.hist(men_ages, bins=len(men_ages), ec="k")



plt.show()

plt.hist(men_ages1, bins=len(men_ages1), ec="k")



plt.show()
women_p_ages = women_p_data[train_data['Survived'] == 1]['Age']

women_p_ages1 = women_p_data[train_data['Survived'] == 0]['Age']



import numpy as np; np.random.seed(13)

import matplotlib.pyplot as plt





plt.hist(women_p_ages, bins=len(women_p_ages), ec="k")



plt.show()
plt.hist(women_p_ages1, bins=len(women_p_ages1), ec="k")



plt.show()
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
train_data[['Parch', 'Survived']].groupby(['Parch']).mean()
women_p_data[['Parch', 'Survived']].groupby(['Parch']).mean()
men_data[['Parch', 'Survived']].groupby(['Parch']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

men_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female':

        if row['Pclass'] == 1 or row['Pclass'] == 2:

            predictions.append(1)

        else:

            if row['Age'] <= 17:

                predictions.append(1)

            else:

                predictions.append(0)

            

    else:

        if row['Pclass'] == 1 and row['Age'] <= 22:

            predictions.append(1)

        else:

            predictions.append(0)

        
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)