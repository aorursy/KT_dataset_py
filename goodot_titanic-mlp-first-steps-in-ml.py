import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.neural_network import MLPRegressor



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
print('Train')

print(train.isnull().sum())

print('==========================')

print('Test')

print(test.isnull().sum())
train = train.fillna({'Age': -0.1})

test = test.fillna({'Age': -0.1})



#encode sex

train['Sex'] = LabelEncoder().fit_transform(train['Sex'])

test['Sex'] = LabelEncoder().fit_transform(test['Sex'])





#encode cabin

train.loc[~train.Cabin.isnull(), 'Cabin'] = 1

train.loc[train.Cabin.isnull(), 'Cabin'] = 0



test.loc[~test.Cabin.isnull(), 'Cabin'] = 1

test.loc[test.Cabin.isnull(), 'Cabin'] = 0





# detect wich is a most common embarking place and fill missed 'Embarked' values with max embarked places

common_embarked = train.groupby(['Embarked'])['Embarked'].value_counts().idxmax()[0]

train = train.fillna({'Embarked': common_embarked})

test = test.fillna({'Embarked': common_embarked})



# fill 'Fare' null values in test

test.loc[test.Fare.isnull(), 'Fare'] = 0
train['Title'] = train.Name.str.split(',', n=1, expand=True)[1].str.split('.',n=1, expand=True)[0]

train['Title'] = train.Title.str.strip()



test['Title'] = test.Name.str.split(',', n=1, expand=True)[1].str.split('.', n=1, expand=True)[0]

test['Title'] = test.Title.str.strip()



train.head()
train.loc[train.Title == 'Ms', 'Title'] = 'Miss'

test.loc[test.Title == 'Ms', 'Title'] = 'Miss'



train.loc[~train.Title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'

test.loc[~test.Title.isin(['Mr', 'Miss', 'Mrs', 'Master']), 'Title'] = 'Other'
train['TicketPrefix'] = train.Ticket.str.split(' ').apply(lambda x: x[0] if len(x) > 1 else 'No')

test['TicketPrefix'] = test.Ticket.str.split(' ').apply(lambda x: x[0] if len(x) > 1 else 'No')

train.head()
train.groupby(['TicketPrefix'])['TicketPrefix'].count()
train.loc[train.TicketPrefix.str.startswith('A'), 'TicketPrefix'] = 'A'

train.loc[train.TicketPrefix.str.startswith('C'), 'TicketPrefix'] = 'C'

train.loc[train.TicketPrefix.str.startswith('F'), 'TicketPrefix'] = 'F'

train.loc[train.TicketPrefix.str.startswith('P'), 'TicketPrefix'] = 'P'

train.loc[train.TicketPrefix.str.startswith('S'), 'TicketPrefix'] = 'S'

train.loc[train.TicketPrefix.str.startswith('W'), 'TicketPrefix'] = 'W'



test.loc[test.TicketPrefix.str.startswith('A'), 'TicketPrefix'] = 'A'

test.loc[test.TicketPrefix.str.startswith('C'), 'TicketPrefix'] = 'C'

test.loc[test.TicketPrefix.str.startswith('F'), 'TicketPrefix'] = 'F'

test.loc[test.TicketPrefix.str.startswith('P'), 'TicketPrefix'] = 'P'

test.loc[test.TicketPrefix.str.startswith('S'), 'TicketPrefix'] = 'S'

test.loc[test.TicketPrefix.str.startswith('W'), 'TicketPrefix'] = 'W'



train.groupby(['TicketPrefix'])['TicketPrefix'].count()
sns.barplot(x='TicketPrefix', y='Survived', data=train)
train['Alone'] = ((train.Parch + train.SibSp) == 0).astype(int)

test['Alone'] = ((test.Parch + test.SibSp) == 0).astype(int)

train.head()
train = train.drop(['Name', 'SibSp', 'Parch', 'Embarked'], axis=1)

test = test.drop(['Name', 'SibSp', 'Parch', 'Embarked'], axis=1)

train.head()
def encode_ticket(t):

    e = {

        'No': 0,

        'A': 1,

        'P': 2,

        'S': 3,

        'C': 4,

        'W': 5,

        'F': 6

    }

    return e.get(t, -1)



train['Ticket'] = train.TicketPrefix.apply(encode_ticket)

test['Ticket'] = test.TicketPrefix.apply(encode_ticket)

train.head()
train.Title = LabelEncoder().fit_transform(train.Title)

test.Title = LabelEncoder().fit_transform(test.Title)

train.head()
train.drop(['TicketPrefix'], axis=1, inplace=True)

test.drop(['TicketPrefix'], axis=1, inplace=True)

train.head()
# combine data

data = pd.concat([train, test])

data.drop(['Survived'], axis=1, inplace=True)

data.head()
data.drop(['PassengerId'], axis=1, inplace=True)
predictors = data[data.Age > 0]

predictors.drop(['Age'], axis=1, inplace=True)

targets = np.array(data[data.Age > 0].Age)

predictors.shape, targets.shape
predictors.head()
predictors = StandardScaler().fit_transform(predictors)
mlp = MLPRegressor(hidden_layer_sizes=(150, 100))

mlp.fit(predictors, targets)

y_pred = mlp.predict(predictors)

mean_squared_error(y_pred, targets), mlp
real_data = np.sort(targets)

predicted_data = np.sort(mlp.predict(predictors))



fig, ax = plt.subplots()

fig.set_size_inches(12, 10)



plt.plot(np.linspace(start=0, stop=len(real_data)*100, num=len(real_data)), real_data, color='b', label='Real Data')

plt.plot(np.linspace(start=0, stop=len(real_data)*100, num=len(real_data)), predicted_data, color='g', label='Predicted Data')



plt.legend()
train.loc[train.Age < 0, 'Age'] = mlp.predict(StandardScaler().fit_transform(train[train['Age'] < 0][['Alone', 'Cabin', 'Fare',  'Pclass', 'Sex','Ticket', 'Title']]))
test.loc[test.Age < 0, 'Age'] = mlp.predict(StandardScaler().fit_transform(test[test['Age'] < 0][['Alone', 'Cabin', 'Fare',  'Pclass', 'Sex','Ticket', 'Title']]))
test.loc[test.Age < 0, 'Age'] = 0.1

train.loc[train.Age < 0, 'Age'] = 0.1



train.head()
predictors = train.drop(['PassengerId', 'Survived'], axis=1)

targets = train[['Survived']]

predictors = StandardScaler().fit_transform(predictors)
x_train, x_test, y_train, y_test = train_test_split(predictors, targets, test_size = 0.05, random_state = 0)
mlp = MLPRegressor(batch_size=50, hidden_layer_sizes=(140))

mlp.fit(x_train, y_train)



y_pred = mlp.predict(x_train).round()

test_pred = mlp.predict(x_test).round()



score = accuracy_score(y_train, y_pred)

test_score = accuracy_score(y_test, test_pred)



mlp, score, test_score
ids = test['PassengerId']

predictions = np.abs(mlp.predict(StandardScaler().fit_transform(test.drop(['PassengerId'], axis=1))).round()).astype(int)



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)