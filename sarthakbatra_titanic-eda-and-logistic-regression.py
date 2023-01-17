# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading in the Test Data

df = pd.read_csv('../input/train.csv')

df.head(2)
# Let's find out the shape/size of our dataset

df.shape
# Let's find out the data types and number of null values for each column/feature

df.info()
df[df['Embarked'].isna()]
df['Age'].plot(kind='hist')
df['Age'].describe()
# Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

fig = plt.figure(figsize=(18,6))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((2,3), (0,0))

df['Survived'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Survival Distribution (Normalized)')



plt.subplot2grid((2,3), (0,1))

df['Sex'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Gender Distribution (Normalized)')



plt.subplot2grid((2,3), (0,2))

df['Pclass'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Pclass (Normalized)')



plt.subplot2grid((2,3), (1,0))

df['SibSp'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('# of Siblings/Spouses (Normalized)')



plt.subplot2grid((2,3), (1,1))

df['Parch'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('# of Parents/Children (Normalized)')



plt.subplot2grid((2,3), (1,2))

df['Embarked'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Embarked')

plt.show()
fig = plt.figure(figsize=(18,15))

plt.subplots_adjust(hspace=0.5)



plt.subplot2grid((4,3), (0,0))

male = df[df['Sex']=='male']

male['Survived'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Survival for Men')



plt.subplot2grid((4,3), (0,1))

female = df[df['Sex']=='female']

female['Survived'].value_counts(normalize=True, ascending=True).plot(kind='bar', alpha=0.7)

plt.title('Survival for Women')



plt.subplot2grid((4,3), (0,2))

df['Sex'][df['Survived']==1].value_counts(normalize=True, ascending=True).plot(kind='bar', alpha=0.7)

plt.title('Survival based on Gender')



plt.subplot2grid((4,3), (1,0), colspan=2)

for ticketclass in sorted(df['Pclass'].unique()):

    df['Age'][df['Pclass']==ticketclass].plot(kind='kde')

plt.legend(('1st','2nd','3rd'))

plt.title('Ticket Class and Age')



ax = plt.subplot2grid((4,3), (1,2))

df.groupby(['Survived', 'Pclass']).size().unstack().plot(kind='bar', stacked=True, ax=ax)

plt.title('Ticket Class and Survival')



plt.subplot2grid((4,3), (2,0))

df['Survived'][(df['Sex']=='female') & (df['Pclass']==1)].value_counts(normalize=True, ascending=True).plot(kind='bar', alpha=0.7)

plt.title('Rich Women Survival')



plt.subplot2grid((4,3), (2,1))

df['Survived'][(df['Sex']=='male') & (df['Pclass']==3)].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Poor Men Survival')



ax = plt.subplot2grid((4,3), (2,2))

df.groupby(['Parch', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)

plt.title('Family and Survival: Parent/Child')



plt.subplot2grid((4,3), (3,0), colspan=2)

for survived in sorted(df['Survived'].unique()):

    df['Age'][df['Survived']==survived].plot(kind='kde')

plt.legend(('Died','Survived'))

plt.title('Survival and Age')



ax = plt.subplot2grid((4,3), (3,2))

df.groupby(['SibSp', 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)

plt.title('Family and Survival: Sibling/Spouse')



plt.show()
# Extract the titles from names

titles = df['Name'].str.extract(' ([A-za-z]+)\.', expand=False)



# Create a dataframe of titles and survival

titles_df = pd.DataFrame()

titles_df['Title'] = titles

titles_df['Survived'] = df['Survived']
# Have a look at the raw numbers

titles_df.groupby(['Title', 'Survived']).size().unstack().T
titles_df['Title'].value_counts().head(6)
mask_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Rev', 'Dr']

titles_df = titles_df.loc[titles_df['Title'].isin(mask_titles), :]

titles_df.groupby(['Title', 'Survived']).size().unstack().plot(kind='bar')

plt.show()
age_eda = df.groupby(['Survived', 'Age']).size().rename('count').reset_index()
sns.catplot(x="Survived", y="Age", hue="Survived", kind="violin", inner="stick", data=age_eda);
del age_eda

age_eda = df.copy()

age_eda['Age'].dropna(inplace=True)

age_eda['Age'] = age_eda['Age'].astype(int)

age_eda.loc[ age_eda['Age'] <= 11, 'Age'] = 0

age_eda.loc[(age_eda['Age'] > 11) & (age_eda['Age'] <= 18), 'Age'] = 1

age_eda.loc[(age_eda['Age'] > 18) & (age_eda['Age'] <= 22), 'Age'] = 2

age_eda.loc[(age_eda['Age'] > 22) & (age_eda['Age'] <= 27), 'Age'] = 3

age_eda.loc[(age_eda['Age'] > 27) & (age_eda['Age'] <= 32), 'Age'] = 4

age_eda.loc[(age_eda['Age'] > 32) & (age_eda['Age'] <= 40), 'Age'] = 5

age_eda.loc[(age_eda['Age'] > 40), 'Age'] = 6
age_eda.groupby(['Age', 'Survived']).size().unstack().plot(kind='bar')
# Imports for learning algorithms

from sklearn import linear_model

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_test = train.append(test, sort=False)



train_test.reset_index(inplace=True)

train_test.drop(['index'], inplace=True, axis=1)

train_test.head(2)
def add_titles(data):

    # Extract titles from Name section

    data['Title'] = data['Name'].str.extract(' ([A-za-z]+)\.', expand=False)

    

    # Title dictionary based on EDA, key 5 effectively translates to 'other title'

    title_mapping = {"Mr": 'Mr',

                     "Miss": 'Miss',

                     "Mrs": 'Mrs', 

                     "Master": 'Master',

                     "Rev": 'Rev',

                     "Dr": 'Other',

                     "Col": 'Other',

                     "Major": 'Other', 

                     "Mlle": 'Other',

                     "Countess": 'Other',

                     "Ms": 'Other',

                     "Lady": 'Other',

                     "Jonkheer": 'Other',

                     "Don": 'Other',

                     "Dona" : 'Other',

                     "Mme": 'Other',

                     "Capt": 'Other',

                     "Sir": 'Other'}

    

    data['Title'] = data['Title'].map(title_mapping)
add_titles(train_test)
def clean_fare(data):

    # Fare in the training set has no missing values but test set does

    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())    
clean_fare(train_test)
fig, ax = plt.subplots(figsize=(8,6))

age_grouped = train_test.iloc[:891].groupby(['Sex','Pclass','Title'])

age_grouped['Age'].median().unstack().plot(kind = 'bar', ax=ax)

plt.show()
age_grouped = train_test.iloc[:891].groupby(['Sex','Pclass','Title'])

age_grouped = age_grouped.median()

age_grouped = age_grouped.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

age_grouped.head(3)
def clean_age(data):

    data["Age"] = data.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
clean_age(train_test)
train_test.loc[train_test['Age'].isnull()]
train_test.loc[979, 'Age'] = train_test['Age'].median()
def bin_age(dataset):

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 32), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
bin_age(train_test)
def clean_embarked(data):

    # Fill missing values with most common embarkment point

    data['Embarked'] = data['Embarked'].fillna('S')
clean_embarked(train_test)
def clean_cabin(data):

    # Fill na values with 'Unknown' or simply 'U'

    data['Cabin'].fillna('U', inplace=True)

    data['Cabin'] = data['Cabin'].map(lambda x: x[0])
clean_cabin(train_test)
def clean_family(data):

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
clean_family(train_test)
# Add label and one-hot encoding for categorical lables



def encode(data, labels):

    for label in labels:

        data = data.join(pd.get_dummies(data[label], prefix = label))

        data.drop(label, axis=1, inplace=True)

    return data
store = train_test.copy()
train_test = encode(train_test, ['Pclass', 'Sex', 'Embarked', 'Title', 'Cabin'])

train_test.head(1).T
train = train_test.loc[:890, :]

test = train_test.loc[891:, :]
def model(classifier, train, test):

    target = train['Survived'].values

    features = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis = 1).values

    

    scores = cross_val_score(classifier, features, target, cv=5)

    print(f'Scores for 5 fold CV: {round(np.mean(scores*100))}')

    

    classifier_ = classifier.fit(features, target)

    # print(classifier_.score(X_test, y_test))

    test = test.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis = 1)

    predictions = classifier_.predict(test).astype(int)

    

    return predictions
def submit(predictions):

    submission = pd.read_csv('../input/gender_submission.csv')

    submission['Survived'] = predictions

    submission.to_csv('submission.csv', index=False)

    return submission
classifier = linear_model.LogisticRegression(solver='liblinear')

predictions = model(classifier, train, test)
classifier = SVC(gamma='auto', kernel='linear')

predictions = model(classifier, train, test)
classifier = DecisionTreeClassifier(random_state = 1, max_depth = 3)

predictions = model(classifier, train, test)
classifier = GradientBoostingClassifier()

predictions = model(classifier, train, test)
classifier = KNeighborsClassifier(3)

predictions = model(classifier, train, test)
classifier = GaussianNB()

predictions = model(classifier, train, test)
classifier = AdaBoostClassifier()

predictions = model(classifier, train, test)
classifier = RandomForestClassifier(n_estimators=100, max_depth = 7)

predictions = model(classifier, train, test)
xg_test = test.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis = 1).as_matrix()

xg_train = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis = 1).as_matrix()

target = train['Survived'].values





classifier = XGBClassifier()

classifier_ = classifier.fit(xg_train, target)

classifier_.score(xg_train, target)
predictions = classifier_.predict(xg_test).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)
submission = submit(predictions)

submission['Survived'].value_counts()