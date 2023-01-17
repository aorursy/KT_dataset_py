import numpy as np

import pandas as pd

import os

import glob

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

print(os.listdir("../input"))
class TextUtils(object):



    def run_sql(self, query, con):

        return pd.read_sql(query, con)



    def unicode_normalize_text(self, form, text):

        return normalize(form, text)



    def strip_start_end_whitespace(self, text):

        return text.strip()



    def lowercase_text(self, text):

        return text.lower()



    def text_length(self, text):

        return len(text) - text.count(' ')



    def count_punctuation(self, text):

        count = sum([1 for char in text if char in string.punctuation])

        return str(round(count/(len(text) - text.count(" ")), 3)*100)



    def df_count_words(self, series):

        all_words = []

        for line in list(df['hashtags']):

            words = line.split(', ')

            for word in words:

                all_words.append(word)

        counter = Counter(all_words).most_common(10)

		# counter = sorted(Counter(all_words).elements())

        print(counter)



    def remove_all_punctuation(self, text):# Obsolete

        return text.translate(string.punctuation)

    

tu = TextUtils()
titanic_converter = {

    'Name': tu.lowercase_text

}

# save PassengerId for final submission

# passengerId = test.PassengerId

# print(passengerId)

# # merge train and test

# # titanic = train.append(test, ignore_index=True)

# create indexes to separate data later on

# train_idx = len(train)

# test_idx = len(titanic) - len(test)

# print(train_idx)

# create indexes to separate data later on

# train_idx = len(train)

# test_idx = len(titanic) - len(test)



train = pd.read_csv('../input/train.csv', converters=titanic_converter, index_col='PassengerId')

test = pd.read_csv('../input/test.csv', converters=titanic_converter, index_col='PassengerId')

df = pd.concat([train, test], axis = 0, sort=True)#.set_index('PassengerId')

df['Survived'] = df['Survived'].fillna(9).astype(int)

df.tail()
one_ = df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

sns.countplot(x='Pclass', data=one_, palette='Set3')

print(one_)

	# print(x[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# print(x[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# print(x[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# # g = sns.FacetGrid(x, col='Survived')

	# # g.map(plt.hist, 'Age', bins=20)



	# # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

	# grid = sns.FacetGrid(x, col='Survived', row='Pclass', height=2.2, aspect=1.6)

	# grid.map(plt.hist, 'Age', alpha=.5, bins=20)

	# grid.add_legend()



	# # grid = sns.FacetGrid(train_df, col='Embarked')

	# grid = sns.FacetGrid(x, row='Embarked', height=2.2, aspect=1.6)

	# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

	# grid.add_legend()



	# # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

	# grid = sns.FacetGrid(x, row='Embarked', col='Survived', height=2.2, aspect=1.6)

	# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

	# grid.add_legend()

	# plt.show()
sns.set(style="darkgrid")

one_ = df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

sns.countplot(x='Pclass', data=one_, palette='Set3')

print(one_)

	# print(x[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# print(x[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# print(x[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

	# # g = sns.FacetGrid(x, col='Survived')

	# # g.map(plt.hist, 'Age', bins=20)



	# # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

	# grid = sns.FacetGrid(x, col='Survived', row='Pclass', height=2.2, aspect=1.6)

	# grid.map(plt.hist, 'Age', alpha=.5, bins=20)

	# grid.add_legend()



	# # grid = sns.FacetGrid(train_df, col='Embarked')

	# grid = sns.FacetGrid(x, row='Embarked', height=2.2, aspect=1.6)

	# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

	# grid.add_legend()



	# # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

	# grid = sns.FacetGrid(x, row='Embarked', col='Survived', height=2.2, aspect=1.6)

	# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

	# grid.add_legend()

	# plt.show()

    

#Chart1

one_ = df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

sns.countplot(x='Pclass', data=one_, palette='Set3')

print(one_)

#Chart2

sns.countplot(x='Survived',data=df)
# average_age = train_test["Age"].mean()

# std_age = train_test["Age"].std()

# count_nan_age = train_test["Age"].isnull().sum()

# rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# train_test['Age'][np.isnan(train_test['Age'])] = rand_1

# train_test['Age'] = train_test['Age'].astype(int)



df['Age'] = df['Age'].fillna(0)



female_median_age = df[df['Sex'] == 'female']['Age'].median()

df.loc[(df['Age'] == 0) & (df['Sex'] == 'female'), 'Age'] = female_median_age



male_median_age = df[df['Sex'] == 'male']['Age'].median()

df.loc[(df['Age'] == 0) & (df['Sex'] == 'male'), 'Age'] = male_median_age

df['Age'] = df['Age'].astype(int)

df.head()
def clean_cabin(x):

	try:

		return x[0]

	except TypeError:

		return 'U'



df['Cabin'] = df['Cabin'].apply(clean_cabin)

cabin_map = {'U': 1, 'C': 2, 'B': 3, 'D': 4, 'E': 5, 'A': 6, 'F': 7, 'G': 8, 'T': 9}

df['Cabin'] = df['Cabin'].map(cabin_map)

# df['Cabin'].value_counts()
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().index[0])

df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

# df['Embarked'].unique()
# train_test["Fare"] = train_test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# df[df['Fare'].isnull()]

median_3class_price = df.loc[(df['Sex'] == 'male') & (df['Pclass'] == 3), 'Fare'].median()

df['Fare'] = round(df['Fare'].fillna(median_3class_price), 0).astype(int)
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# train_test['Title'] = train_test['Title'].replace('Mlle', 'Miss')

# train_test['Title'] = train_test['Title'].replace('Ms', 'Miss')

# train_test['Title'] = train_test['Title'].replace('Mme', 'Mrs')

title_map = {'mr': 1,'mrs': 2,'miss': 3,'master': 4,'don': 5,'rev':6,'dr': 7,'major': 8,\

			'lady': 9,'sir': 10,'col': 11,'capt': 12,'countess': 13,'jonkheer': 14, 'dona': 15, 'mlle': 16, 'ms': 17, 'mme': 18}

df['Title'] = df['Title'].map(title_map)
# THE +1 INCLUDES THE PASSENGER. SO IF FAILYSIZE IS == 1 THEN THEY ARE ALONE

# train_test['FamilySize'] = train_test.Parch + train_test.SibSp + 1

# train_test['IsAlone'] = 0

# train_test.loc[train_test['FamilySize'] == 1, 'IsAlone'] = 1



df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

# df.loc[df['FamilySize'] > 1, 'IsAlone'] = 1

# df.loc[df['FamilySize'] == 0, 'IsAlone'] = 0

# df = df.drop(['Parch', 'SibSp'], axis=1)

# df.info()
df['Sex'] = df['Sex'].map({'male': 1, 'female': 2})

# df.head()
df['Ticket'].nunique()
df['ticket'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False).fillna('U')

ticket_map = {'A': 1, 'PC': 2, 'STON': 3, 'PP': 4, 'C': 5, 'SC': 6, 'S': 7, \

              'CA': 8, 'SO': 9, 'W': 10, 'SOTON': 11, 'Fa': 12, 'LINE': 13, \

              'F': 14, 'SW': 15, 'SCO': 16, 'P': 17, 'WE': 18, 'AQ': 19, 'LP': 20, 'U': 21}

df['ticket'] = df['ticket'].map(ticket_map)

df = df.drop(['Ticket', 'Name'], axis=1)

# df['ticket'].unique()
train = df[df['Survived'].notnull()]

test = df[df['Survived'].isnull()].drop('Survived', axis=1)

target = train.pop('Survived')

# X_train = train.drop('Survived', axis=1)

# X_test = test.drop('PassengerId', axis=1).copy()

# Y_train = train["Survived"]

# X_train = titanic_df.drop("Survived",axis=1)

# Y_train = titanic_df["Survived"]

# X_test  = test_df.drop("PassengerId",axis=1).copy()

train.head()

# test.head()
train.head()

# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(train, test)



Y_pred = random_forest.predict(target)



random_forest.score(train, train)
svc = SVC()



svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)



svc.score(X_train, Y_train)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LogisticRegression



# # add your data here

# X_train,y_train = make_my_dataset()



# # it takes a list of tuples as parameter

# pipeline = Pipeline([

#     ('scaler',StandardScaler()),

#     ('clf', LogisticRegression())

# ])



# # use the pipeline object as you would

# # a regular classifier

# pipeline.fit(X_train,y_train)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# submission = pd.DataFrame({

#         "PassengerId": test_df["PassengerId"],

#         "Survived": Y_pred

#     })

# submission.to_csv('titanic.csv', index=False)