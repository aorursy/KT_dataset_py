import numpy as np

import pandas as pd

import os

from keras import models, layers, Sequential

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
## A merged dataset for common manipulations

merged_df = pd.concat((train_df.drop(['Survived'], axis = 1), test_df))

merged_df.head()
print(train_df.shape)

print(test_df.shape)

print(merged_df.shape)
train_df.info()
import seaborn as sb

from matplotlib import pyplot as plt
train_df.groupby(['Pclass'])['Survived'].sum() / train_df.groupby(['Pclass'])['Survived'].count()
sb.barplot(x = 'Pclass', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("Pclass")

plt.show()
train_df.groupby(['Sex'])['Survived'].sum() / train_df.groupby(['Sex'])['Survived'].count()
sb.barplot(x = 'Sex', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("Sex")

plt.show()
train_df.groupby(['SibSp'])['Survived'].sum() / train_df.groupby(['SibSp'])['Survived'].count()
sb.barplot(x = 'SibSp', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("SibSp")

plt.show()
train_df.groupby(['Parch'])['Survived'].sum() / train_df.groupby(['Parch'])['Survived'].count()
sb.barplot(x = 'Parch', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("Parch")

plt.show()
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
train_df.groupby(['Family'])['Survived'].sum() / train_df.groupby(['Family'])['Survived'].count()
sb.barplot(x = 'Family', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("Family")

plt.show()
train_df.drop(['Family'], axis = 1, inplace = True) #Dropping it here. We will use it as a custom feature later
ticket_num_records = train_df.groupby(['Ticket']).size().sort_values(ascending=False).to_dict()

train_df.groupby(['Ticket']).size().sort_values(ascending=False).head()
train_df['Companion'] = train_df['Ticket'].apply(lambda x: ticket_num_records[x])
train_df.groupby(['Companion'])['Survived'].sum() / train_df.groupby(['Companion'])['Survived'].count()
sb.barplot(x = 'Companion', y = 'Survived', data = train_df)

plt.ylabel("Survived")

plt.xlabel("Companion")

plt.show()
train_df.drop(['Companion'], axis = 1, inplace = True)
train_df.groupby(pd.cut(train_df["Fare"], np.arange(0, 350, 25)))['Survived'].sum() / train_df.groupby(pd.cut(train_df["Fare"], np.arange(0, 350, 25)))['Survived'].count()
train_df["Cabin"].unique()
train_df['CabinId'] = train_df['Cabin'].apply(lambda x: 'None' if pd.isna(x) else x[0])

train_df.groupby(['CabinId'])['Survived'].sum() / train_df.groupby(['CabinId'])['Survived'].count()
train_df.drop(['CabinId'], axis = 1, inplace = True)
train_df['Embarked'].unique()
train_df.groupby(['Embarked'])['Survived'].sum() / train_df.groupby(['Embarked'])['Survived'].count()
import re



train_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0]).unique()
train_df['Title'] = train_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0])

train_df.groupby(['Title'])['Survived'].sum() / train_df.groupby(['Title'])['Survived'].count()
train_df.groupby(['Title'])['Survived'].count()
train_df.drop(['Title'], axis = 1, inplace = True)
np.nanmean(train_df['Age'])
np.nanmean(train_df[train_df['Name'].str.contains('Master')]['Age'])
np.nanmean(train_df[train_df['Name'].str.contains('Miss')]['Age'])
#Adding Title as a feature

merged_df['Title'] = merged_df['Name'].apply(lambda x: re.compile('.+?[,][\s](.*?)[\.][\s].+').findall(x)[0])
merged_df.head()
#We will take the mean values from training data, as per convention



boymean = np.nanmean(train_df[train_df['Name'].str.contains('Master.')]['Age'])

girlmean = np.nanmean(train_df[train_df['Name'].str.contains('Miss.')]['Age'])

meanage = np.nanmean(train_df['Age'])
merged_df['Age'] = np.where(np.isnan(merged_df['Age']) & (merged_df['Title'] == 'Master'), boymean, merged_df['Age'])

merged_df['Age'] = np.where(np.isnan(merged_df['Age']) & (merged_df['Title'] == 'Miss'), girlmean, merged_df['Age'])

merged_df['Age'] = merged_df['Age'].fillna(meanage)
# Replacing the only missing Fare value with mean Fare. Then converting it to a binary feature



merged_df['Fare'] = merged_df['Fare'].fillna(np.nanmedian(merged_df['Fare']))

merged_df['Fare'] = merged_df['Fare'].apply(lambda x: 1 if x > 75.0 else 0)
# Reformatting the Cabin column



merged_df['Cabin'] = merged_df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
# Filling two empty Embarked data with an arbitrary character



merged_df['Embarked'] = merged_df['Embarked'].fillna('N')
merged_df['Family'] = merged_df['Parch'] + merged_df['SibSp']
merged_df.info()
merged_df.head()
# Removing features that I don't think has any significance



merged_df.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis = 1, inplace = True)
merged_df.head()
train_df.groupby(pd.cut(train_df["Age"], np.arange(0, 100, 20)))['Survived'].sum() / train_df.groupby(pd.cut(train_df["Age"], np.arange(0, 100, 20)))['Survived'].count()
maxAge = train_df['Age'].max()

minAge = train_df['Age'].min()

merged_df['Age'] = (merged_df['Age'] - minAge)/(maxAge - minAge)
merged_df.head()
merged_df['Age'].max()
dummiesPclass = pd.get_dummies(merged_df['Pclass'], prefix = 'Pclass')

merged_df = pd.concat([merged_df, dummiesPclass], axis=1)

merged_df.head()
dummiesFare = pd.get_dummies(merged_df['Fare'], prefix = 'Fare')

merged_df = pd.concat([merged_df, dummiesFare], axis=1)

merged_df.head()
merged_df.groupby(['Title'])['PassengerId'].count()
merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Miss' if (x in ['Mlle', 'Mme', 'Ms']) else x)

merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Mrs' if (x in ['Dona', 'Lady']) else x)

merged_df['Title'] = merged_df['Title'].apply(lambda x: 'Mr' if (x == 'Rev') else x)

merged_df['Title'] = merged_df['Title'].apply(lambda x: x if (x in ['Master', 'Mr', 'Mrs', 'Miss']) else 'Other')
merged_df.groupby(['Title'])['PassengerId'].count()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

merged_df['Title'] = le.fit_transform(merged_df['Title'])
dummiesTitle = pd.get_dummies(merged_df['Title'], prefix = 'Title')

merged_df = pd.concat([merged_df, dummiesTitle], axis=1)

merged_df.head()
merged_df['Sex'] = le.fit_transform(merged_df['Sex'])

dummiesSex = pd.get_dummies(merged_df['Sex'], prefix = 'Sex')

merged_df = pd.concat([merged_df, dummiesSex], axis=1)

merged_df.head()
dummiesCabin = pd.get_dummies(merged_df['Cabin'], prefix = 'Cabin')

merged_df = pd.concat([merged_df, dummiesCabin], axis=1)

merged_df.head()
merged_df['Embarked'] = le.fit_transform(merged_df['Embarked'])

dummiesEmbarked = pd.get_dummies(merged_df['Embarked'], prefix = 'Embarked')

merged_df = pd.concat([merged_df, dummiesEmbarked], axis=1)

merged_df.head()
merged_df.head()
merged_df['Family'] = merged_df['Family'].apply(lambda x: 'N' if x == 0 else ('S' if x < 4 else 'L'))
merged_df['Family'] = le.fit_transform(merged_df['Family'])

dummiesFamily = pd.get_dummies(merged_df['Family'], prefix = 'Family')

merged_df = pd.concat([merged_df, dummiesFamily], axis=1)

merged_df.head()
merged_df.drop(['Pclass', 'Sex', 'Fare', 'Cabin', 'Embarked', 'Title', 'Family'], axis = 1, inplace = True)
merged_df.head()
train_df_x = merged_df[:891]

test_df_x = merged_df[891:]
print(test_df_x.shape)

print(test_df.shape)
train_df_y = train_df['Survived']
train_df = train_df_x.copy()

train_df['Survived'] = train_df_y

train_df.drop(['PassengerId'], axis = 1, inplace = True)

train_df.head()
from sklearn.model_selection import train_test_split



def train_and_test(model_specific_tasks, df, it = 20):

    accsum = 0

    minacc = 1.0

    maxacc = 0

    

    for i in range(it):

        print('Iteration: ', (i + 1), end = '\r')

        train, test = train_test_split(df, test_size=0.2)



        train_x = train.drop(['Survived'], axis=1)

        test_x = test.drop(['Survived'], axis=1)



        train_y = train['Survived']

        test_y = test['Survived']



        train_x = np.asarray(train_x).astype('float32')

        train_y = np.asarray(train_y).astype('float32')



        acc = model_specific_tasks(train_x, train_y, test_x, test_y)

        accsum += acc

        minacc = acc if acc < minacc else minacc

        maxacc = acc if acc > maxacc else maxacc

        

    print('Avg. accuracy: ', (accsum / it))

    print('Min. accuracy: ', minacc)

    print('Max. accuracy: ', maxacc)
def logistic_reg(train_x, train_y, test_x, test_y):

    model = LogisticRegression(solver='sag', max_iter=1000)

    model.fit(train_x, train_y)

    return model.score(test_x, test_y)
train_and_test(logistic_reg, train_df.copy(), it = 50)
def rfc(train_x, train_y, test_x, test_y):

    model = RandomForestClassifier(n_estimators=100)

    model.fit(train_x, train_y)

    return model.score(test_x, test_y)
train_and_test(rfc, train_df.copy())
def nn(train_x, train_y, test_x, test_y):

    model = Sequential()

    model.add(layers.Dense(32, activation='relu', input_shape = (22,)))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(8, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    

    model.fit(train_x, train_y, epochs=150, batch_size=16, verbose = 0)

    

    return model.evaluate(test_x, test_y, verbose = 0)[1]
train_and_test(nn, train_df.copy(), it = 10)