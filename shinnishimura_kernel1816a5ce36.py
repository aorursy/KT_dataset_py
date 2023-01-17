from sklearn.linear_model import LogisticRegression as LR

from sklearn.svm import SVC

from sklearn.svm import LinearSVC as LSVC

from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.naive_bayes import GaussianNB as GNB

from sklearn.linear_model import Perceptron as P

from sklearn.linear_model import SGDClassifier as SGDC

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.ensemble import AdaBoostClassifier as ABC

from sklearn.ensemble import AdaBoostRegressor as ABR

from sklearn.ensemble import BaggingClassifier as BC

from sklearn.ensemble import StackingClassifier as SC

from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.ensemble import ExtraTreesClassifier as ETC

from sklearn.ensemble import ExtraTreesRegressor as ETR



from sklearn.model_selection import train_test_split as TTS



import math

import random

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.loc[630, 'Age'] = 42

train_data['Survived1'] = train_data['Survived'].apply(lambda x: math.nan if x==0 else 1)

train_data['Survived0'] = train_data['Survived'].apply(lambda x: math.nan if x==1 else 1)
def get_name_info(name):

    surname = ''

    title = ''

    stage = 'surname'

    for letter in name: 

        if letter != ' ': 

            if stage == 'surname': 

                if letter == ',': 

                    stage = 'title'

                else: 

                    surname += letter

            elif stage == 'title': 

                if letter == '.': 

                    break

                else: 

                    title += letter

    return surname, title
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

for data in (train_data, test_data):

    data['Embarked'] = data['Embarked'].fillna('U')

    data['Fare'] = data['Fare'].fillna(0)

    data['Cabin'] = data['Cabin'].fillna('U')

    data['Cabin'] = data['Cabin'].apply(lambda x: 'U' if x=='T' else x[0])

    data['Surname'] = data['Name'].apply(lambda x: get_name_info(x)[0])

    data['Title'] = data['Name'].apply(lambda x: get_name_info(x)[1])
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

title_dict = {'Don': 'Noble', 'Dona': 'Noble', 'Jonkheer': 'Noble', 'Lady': 'Noble', 'Sir': 'Noble', 'theCountess': 'Noble', 

              'Capt': 'Army', 'Major': 'Army', 'Col': 'Army', 

              'Miss': 'Miss', 'Mlle': 'Miss', 'Ms': 'Miss',  

              'Mrs': 'Mrs', 'Mme': 'Mrs', 

              'Mr': 'Mr', 'Dr': 'Dr', 'Rev': 'Rev', 'Master': 'Master'}

cabin_luck = {x: (train_data[['Survived1', 'Cabin']].dropna(axis=0)).groupby('Cabin').size()[x] / (train_data[['Cabin']]).groupby('Cabin').size()[x] for x in full_data['Cabin'].unique()}

class_luck = {x: (train_data[['Survived1', 'Pclass']].dropna(axis=0)).groupby('Pclass').size()[x] / (train_data[['Pclass']]).groupby('Pclass').size()[x] for x in full_data['Pclass'].unique()}

embark_luck = {x: (train_data[['Survived1', 'Embarked']].dropna(axis=0)).groupby('Embarked').size()[x] / (train_data[['Embarked']]).groupby('Embarked').size()[x] for x in full_data['Embarked'].unique()}

unique_ticket = {x: (full_data[['Ticket']]).groupby('Ticket').size()[x] for x in full_data['Ticket'].unique()}
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

for data in (train_data, test_data):

    data['NoAge'] = data['Age'].apply(lambda x: 1 if math.isnan(x) else 0)

    data['Free'] = data['Fare'].apply(lambda x: 1 if x==0 else 0)

    data['Title'] = data['Name'].apply(lambda x: get_name_info(x)[1])

    data['TitleType'] = data['Title'].apply(lambda x: title_dict[x])

    data['CabinLuck'] = data['Cabin'].apply(lambda x: cabin_luck[x])

    data['ClassLuck'] = data['Pclass'].apply(lambda x: class_luck[x])

    data['EmbarkLuck'] = data['Embarked'].apply(lambda x: embark_luck[x])

    data['UniqueTicket'] = data['Ticket'].apply(lambda x: unique_ticket[x])
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

for title in full_data['TitleType'].unique():

    full_data['Title'+title] = full_data['TitleType'].apply(lambda x: math.nan if x != title else 1)

title_mean_age = {x: ((full_data[['Age', 'Title'+x]].dropna(axis=0))['Age'].sum() / (full_data[['Age', 'TitleType']].dropna(axis=0)).groupby('TitleType').size()[x]) for x in full_data['TitleType'].unique()}

title_luck = {x: 1 - (train_data[['Survived0', 'TitleType']].dropna(axis=0)).groupby('TitleType').size()[x] / (train_data[['TitleType']]).groupby('TitleType').size()[x] for x in full_data['TitleType'].unique()}

ticket_luck = {x: 1 - (train_data[['Survived0', 'UniqueTicket']].dropna(axis=0)).groupby('UniqueTicket').size()[x] / (train_data[['UniqueTicket']]).groupby('UniqueTicket').size()[x] for x in full_data['UniqueTicket'].unique()}
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

for data in (train_data, test_data):

    data['Age'] = data['Age'].fillna(data['TitleType'].apply(lambda x: title_mean_age[x]))

    data['AgeLog'] = np.log(data['Age']).apply(lambda x: 0 if x==-math.inf else x)

    data['TitleTypeLuck'] = data['TitleType'].apply(lambda x: title_luck[x])

    data['TicketLuck'] = data['UniqueTicket'].apply(lambda x: ticket_luck[x])

    

    data['FareCapita'] = data['Fare'] / data['UniqueTicket']

    data['FareCapitaLog'] = np.log(data['FareCapita']).apply(lambda x: 0 if x==-math.inf else x)

    data['FareLog'] = np.log(data['Fare']).apply(lambda x: 0 if x==-math.inf else x)

    data['FareLogCapita'] = data['FareLog'] / data['UniqueTicket']

    

    data['Infant'] = data['Age'].apply(lambda x: 0 if x>1.7 else 1)

    data['Child'] = data['Age'].apply(lambda x: 0 if 1.7>=x and x<15 else 1)

    

    data['Pclass'] = data['Pclass'].astype(str)
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

item = 'FareLogCapita'

print("Skewness: %f" % full_data[item].skew())

print("Kurtosis: %f" % full_data[item].kurt())

sns.distplot(full_data[item])

res = stats.probplot(full_data[item], plot=plt)

full_data[[item, 'Survived']].describe()

sns.catplot(data=full_data, x='Survived', y='AgeLog', hue='Sex', kind='swarm', height=5, aspect = 2)
train_data.columns
full_data = pd.concat([train_data, test_data], sort=True, axis=0)

# Droppable Pre-Dummy

#       ['Age', 'AgeLog', 'Cabin', 'CabinLuck', 'Child', 'ClassLuck',

#        'EmbarkLuck', 'Embarked', 'Fare', 'FareCapita', 'FareCapitaLog',

#        'FareLog', 'FareLogCapita', 'Free', 'Infant', 'Name', 'NoAge', 'Parch',

#        'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Surname', 'Survived',

#        'Survived0', 'Survived1', 'Ticket', 'TicketLuck', 'Title', 'TitleTypeLuck',

#        'TitleType', 'UniqueTicket']

dropPreDummy = [

    'Survived0', 'Survived1', 'Name', 'Surname', 'Ticket', 'FareLog', 

    'Title', 'Fare', 'FareCapita', 'TicketLuck', 

    'Embarked', 'Cabin', 'FareCapitaLog', 'Parch', 'SibSp',

]

dropPostDummy = [

    

]

data = pd.get_dummies(full_data.drop(dropPreDummy+['PassengerId'], axis=1)).astype(float).corr()

plt.figure(figsize = (20, 20))

sns.heatmap(data=data, annot=True, fmt = ".2f", cmap = "seismic", square = True, vmax = 1.0, vmin = -1.0)
seed = 2

price = 'Survived'

full_data = pd.concat([train_data, test_data], sort = True, axis=0)

y = train_data[price]

X = pd.get_dummies(full_data.drop(dropPreDummy, axis=1)).dropna(axis=0).drop([price, 'PassengerId']+dropPostDummy, axis=1)

train_X, val_X, train_y, val_y = TTS(X, y, random_state=seed, test_size=(418/1309))



model = LR()

try:

    model = model(random_state=seed)

except TypeError:

    pass



model.fit(train_X, train_y)

prediction = model.predict(val_X)

mean = MAE(val_y, prediction)

model.fit(val_X, val_y)

print(str((1-mean)*100)+'% correct')



full_data[price] = full_data[price].apply(lambda x: 1 if math.isnan(x) else math.nan)

X_test = pd.get_dummies(full_data.drop(dropPreDummy, axis=1)).dropna(axis=0).drop([price, 'PassengerId'], axis=1)



try:

    importance, values = model.feature_importances_, X.columns.values

    plt.figure(figsize = (30, 30))

    plot = pd.DataFrame({'Importance': importance, 'Value': values})

    plot = plot.sort_values(by=['Importance'])

    sns.barplot(y = 'Importance', x = 'Value', data = plot)

except AttributeError: 

    pass



predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, price: predictions})

output.to_csv('submission.csv', index=False)

print(output.head(20))

print('etc')

print('--------------------------')

print("success")