import string, math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

from sklearn.svm import SVC

from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.preprocessing import StandardScaler, MinMaxScaler
print('Last update on', pd.to_datetime('now'))
def load_sanitize(url):

    df = pd.read_csv(url)

    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1}).astype(int)

    #df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)

    df['Cabin'] = df['Cabin'].fillna('None')

    df['Embarked'] = df['Embarked'].fillna(2).replace({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    df['Fare'] = df['Fare'].fillna(0)

    return df
train = load_sanitize('../input/titanic/train.csv')

train.head()
train.info()
test = load_sanitize('../input/titanic/test.csv')

test.head()
train.dtypes
def build_titles(row):

    name = row['Name']

    for title in ['Mrs','Mr','Master','Miss','Major','Rev','Dr', 'Ms','Mlle','Col','Capt','Mme','Countess','Don','Jonkheer']:

        if str.find(name, title) != -1:

            if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']: return 0

            elif title in ['Mrs','Countess', 'Mme']: return 1

            elif title in ['Miss','Mlle', 'Ms']: return 2

            elif title in ['Master']: return 3

            elif title == 'Dr':

                if row['Sex'] == 'Male': return 0

                else: return 1



def nan_ages(row):

    age = row['Age']

    if np.isnan(age):

        if row['Title'] == 0: return train[(train['Title'] == 0)]['Age'].mean()

        elif row['Title'] == 1: return train[(train['Title'] == 1)]['Age'].mean()

        elif row['Title'] == 2: return train[(train['Title'] == 2)]['Age'].mean()

        elif row['Title'] == 3: return train[(train['Title'] == 3)]['Age'].mean()

        elif row['Title'] == 4: return train[(train['Title'] == 4)]['Age'].mean()

    else: return age

    

def build_deck(row):

    cabin = str(row['Cabin'])

    decks = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'None']

    for i, deck in zip(range(0, len(decks)), decks):

        if str.find(cabin, deck) != -1: return i

        

def engineer_features(df):

    df['Title'] = df.apply(build_titles, axis=1)

    df['Deck'] = df.apply(build_deck, axis=1)

    df['Age'] = df.apply(nan_ages, axis=1)

    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['FarePerPassenger'] = df['Fare']/(df['FamilySize'] + 1)

    return df
train = load_sanitize('../input/titanic/train.csv')

train, test = engineer_features(train), engineer_features(test)

train.head(6)
test.head()
def create_bins(df, columns, q=5):

    for column in columns:

        df[column] = pd.qcut(df[column], q).cat.codes



def normalize_data(df, columns):

    minMaxScaler = MinMaxScaler()

    for column in columns:

        df[column] = minMaxScaler.fit_transform(df[[column]])
#create_bins(train, ['Age'])

#create_bins(train, ['Fare','FarePerPassenger'], q=10)

normalize_data(train, ['Pclass','Age','SibSp','Parch','Fare','Embarked','Title','Deck','FamilySize','FarePerPassenger'])

train.head()
#create_bins(test, ['Age'])

#create_bins(test, ['Fare','FarePerPassenger'], q=10)

normalize_data(test, ['Pclass','Age','SibSp','Parch','Fare','Embarked','Title','Deck','FamilySize','FarePerPassenger'])

test.head()
fig, ax = plt.subplots(figsize=(18,12))



s1 = plt.subplot(221)

plt.hist(train['Age'], color='limegreen', alpha=0.75)

plt.title('Passengers by Age', fontsize=14, fontweight='bold', color='#333333')

plt.grid(which='major', axis='y', color='#CCCCCC')

plt.xlabel('Ages')

[s1.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

s1.set_axisbelow(True)





gender = pd.DataFrame(data={'count': [train[(train['Sex'] == i)]['Sex'].count() for i in range(0,2)]}, index=['male','female'])

s2 = plt.subplot(222)

plt.bar(x=gender.index, height=gender['count'], color=['blue','red'], alpha=0.75)

plt.title('Passengers by Gender', fontsize=14, fontweight='bold', color='#333333')

plt.grid(which='major', axis='y', color='#CCCCCC')

plt.xlabel('Gender')

[s2.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

s2.set_axisbelow(True)



title = pd.DataFrame(data={'count': [train[(train['Title'] == i)]['Title'].count() for i in range(0,4)]}, index=['Mr','Mrs','Miss','Master'])

s3 = plt.subplot(223)

#plt.bar(x=title.index, height=title['count'], color=['limegreen','blue','red','purple'], alpha=0.75)

plt.hist(train['Title'], color='purple', alpha=0.75)

plt.title('Passengers by Title', fontsize=14, fontweight='bold', color='#333333')

plt.grid(which='major', axis='y', color='#CCCCCC')

plt.xlabel('Titles')

[s3.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

s3.set_axisbelow(True)



pclass = pd.DataFrame(data={'count': [train[(train['Pclass'] == i)]['Pclass'].count() for i in range(0,4)]}, index=range(0,4))

s4 = plt.subplot(224)

plt.hist(train['Pclass'], color='orange', alpha=0.75)

#plt.bar(x=pclass.index, height=pclass['count'], color=['limegreen','blue','red','purple'], alpha=0.75)

plt.title('Passengers by Class', fontsize=14, fontweight='bold', color='#333333')

plt.grid(which='major', axis='y', color='#CCCCCC')

plt.xlabel('Class')

[s4.spines[spine].set_visible(False) for spine in ('top', 'right', 'left')]

#s4.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

#s4.xaxis.set_ticks(np.arange(0, 4, 1))



s4.set_axisbelow(True)



plt.show();



fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(train.drop('PassengerId',axis=1).corr(), annot=True, vmin=0, cmap=plt.cm.YlGnBu)

plt.title('Features Correlations', fontsize=14, fontweight='bold', color='#333333')

plt.show();
print('Train shape:', train.shape)

print('Test shape:', test.shape)
features = [f for f in test.columns if f not in ['PassengerId','Name','Ticket','Cabin']]



x = train[features]

y = train['Survived']

testing = test[features]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
svc = SVC(gamma='auto', probability=True)

svc.fit(x_train, y_train)

yhat = svc.predict(x_test)

print('Train score: %.3f \nTest score: %.3f ' % (svc.score(x_train,y_train), svc.score(x_test,yhat)))

print('Cross Validation:',cross_val_score(svc, x, y, cv=5).mean())

print('\n',classification_report(y_test, yhat))
logreg = LogisticRegression(max_iter=300)

logreg.fit(x_train, y_train)

yhat = logreg.predict(x_test)

print('Train score: %.3f \nTest score: %.3f ' % (logreg.score(x_train,y_train), logreg.score(x_test,yhat)))

print('Cross Validation:',cross_val_score(logreg, x, y, cv=5).mean())

print('\n', classification_report(y_test, yhat))
knn = KNeighborsClassifier(n_neighbors=7)  

knn.fit(x_train, y_train)

yhat = knn.predict(x_test)

print('Train score: %.3f \nTest score: %.3f ' % (knn.score(x_train,y_train), knn.score(x_test,yhat)))

print('Cross Validation:',cross_val_score(knn, x, y, cv=5).mean())

print('\n', classification_report(y_test, yhat))
gpc = GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0)

gpc.fit(x_train, y_train)

yhat = gpc.predict(x_test)

print('Train score: %.3f \nTest score: %.3f ' % (gpc.score(x_train,y_train), gpc.score(x_test,yhat)))

print('Cross Validation:',cross_val_score(gpc, x, y, cv=5).mean())

print('\n', classification_report(y_test, yhat))
forest = RandomForestClassifier()

forest.fit(x_train, y_train)

yhat = forest.predict(x_test)

print('Train score: %.3f \nTest score: %.3f ' % (forest.score(x_train,y_train), forest.score(x_test,yhat)))

print('Cross Validation:',cross_val_score(forest, x, y, cv=5).mean())

print('\n', classification_report(y_test, yhat))
for model, name in zip([svc, logreg, knn, gpc, forest],['svc','logreg','knn','gpc','forest']):

    model.fit(x_train, y_train)

    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': model.predict(test[features])})

    submission.to_csv('%s.csv' % name, index=False)