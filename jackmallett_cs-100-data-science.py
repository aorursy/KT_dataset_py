import numpy as np

import pandas as pd

import seaborn as sns
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

sv = train_data[train_data['Survived'] == 1]
train_data[:10]
train_data.describe(include='all')
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
testing = 'Age'


#train_data[testing] = pd.qcut(train_data[testing], 6)
test_data.corr()
train_data.drop('Cabin',axis = 1,inplace = True)
train_data.drop('Ticket',axis = 1,inplace = True)
train_data.drop('Name',axis = 1,inplace = True)
train_data.drop('PassengerId',axis = 1,inplace = True)

train_data.head()
train_data = train_data.fillna({"Embarked": "S"})

train_data = train_data.fillna({"Age": 30})
#train_data['Age'] = pd.cut(train_data['Age'], 6)
wom = train_data[train_data['Sex'] == 'female']
#wom['Age'] = pd.cut(wom['Age'], 6)


womend = wom[wom['Survived'] == 0]



w3 = wom.query('not (Pclass == 3 and Age > 35)')

print(len(w3) / len(test_data))

w3.describe(include='all')

#w3.corr()

#w2 = wom.query('Pclass < 3 and Age < 38')

#w2[['SibSp', 'Survived']].groupby(['SibSp']).mean()

#sns.countplot('SibSp', data = w2)
s = sv.query()






men = train_data[train_data['Sex'] == 'male']

mens = men[men['Survived'] == 1]

#mens.head(len(mens))

#men.corr()

mens[['Age', 'Survived']].groupby(['Age']).mean()



kids = train_data[train_data['Age'] < 25]

kids.corr()

ks = kids[kids['SibSp'] > 3]

#ks.head()

#ks[['Parch', 'Survived']].groupby(['Parch']).mean()
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female' and row['Pclass'] != 3:

        predictions.append(1)

    elif row['Age'] < 16 and row['Pclass'] != 3:

        predictions.append(1)

    else:

        predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)