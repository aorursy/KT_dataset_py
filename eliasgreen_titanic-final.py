import numpy as np

import pandas as pd

import seaborn as sns
sns.set_context('talk')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.describe()
df_train.info()
plt1 = sns.barplot(x="Pclass", y="Survived", data=df_train[['Survived', 'Pclass']])
plt2 = sns.distplot(df_train['Pclass'].dropna())
for index, row in df_train.iterrows():

    df_train.at[index,'Name'] = row['Name'].split(',')[1].split()[0]

    

df_train['Name']
plt1 = sns.barplot(x="Name", y="Survived", data=df_train[['Survived', 'Name']])

settings = plt1.set_xticklabels(plt1.get_xticklabels(),rotation=60, ha="right")
plt2 = sns.countplot(df_train['Name'].dropna())

settings = plt2.set_xticklabels(plt2.get_xticklabels(),rotation=60, ha="right")
plt1 = sns.barplot(x="Sex", y="Survived", data=df_train[['Survived', 'Sex']])
plt2 = sns.countplot(df_train['Sex'].dropna())
plt1 = sns.lineplot(x="Survived", y="Age", data=df_train[['Age', 'Survived']])
plt2 = sns.distplot(df_train['Age'].dropna())
plt1 = sns.lineplot(x="Survived", y="SibSp", data=df_train[['SibSp', 'Survived']])
plt2 = sns.countplot(df_train['SibSp'].dropna())
plt1 = sns.lineplot(x="Survived", y="Parch", data=df_train[['Parch', 'Survived']])
plt2 = sns.countplot(df_train['Parch'].dropna())
plt1 = sns.lineplot(x="Survived", y="Fare", data=df_train[['Fare', 'Survived']])
plt2 = sns.distplot(df_train['Fare'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Cabin'] = df_train['Cabin'].fillna('missing')

le.fit(df_train['Cabin'])

df_train['Cabin'] = le.transform(df_train['Cabin'])



df_train['Cabin']
plt1 = sns.lineplot(x="Survived", y="Cabin", data=df_train[['Cabin', 'Survived']])
plt2 = sns.distplot(df_train['Cabin'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Embarked'] = df_train['Embarked'].fillna('missing')

le.fit(df_train['Embarked'])

df_train['Embarked'] = le.transform(df_train['Embarked'])



df_train['Embarked']
plt1 = sns.lineplot(x="Survived", y="Embarked", data=df_train[['Embarked', 'Survived']])
plt2 = sns.countplot(df_train['Embarked'].dropna())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Name'] = df_train['Name'].fillna('missing')

le.fit(df_train['Name'])

df_train['Name'] = le.transform(df_train['Name'])



df_train['Name']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_train['Sex'] = df_train['Sex'].fillna('missing')

le.fit(df_train['Sex'])

df_train['Sex'] = le.transform(df_train['Sex'])



df_train['Sex']
df_train['Age'].fillna((df_train['Age'].mean()), inplace=True)
for index, row in df_test.iterrows():

    df_test.at[index,'Name'] = row['Name'].split(',')[1].split()[0]

    

df_test['Name']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_test['Name'] = df_test['Name'].fillna('missing')

le.fit(df_test['Name'])

df_test['Name'] = le.transform(df_test['Name'])



df_test['Name']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_test['Cabin'] = df_test['Cabin'].fillna('missing')

le.fit(df_test['Cabin'])

df_test['Cabin'] = le.transform(df_test['Cabin'])



df_test['Cabin']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_test['Embarked'] = df_test['Embarked'].fillna('missing')

le.fit(df_test['Embarked'])

df_test['Embarked'] = le.transform(df_test['Embarked'])



df_test['Embarked']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df_test['Sex'] = df_test['Sex'].fillna('missing')

le.fit(df_test['Sex'])

df_test['Sex'] = le.transform(df_test['Sex'])



df_test['Sex']
df_test['Age'].fillna((df_test['Age'].mean()), inplace=True)

df_test['Fare'].fillna((df_test['Fare'].mean()), inplace=True)
from sklearn.ensemble import GradientBoostingClassifier

clf1 = GradientBoostingClassifier(criterion='friedman_mse', init=None,

                           learning_rate=0.1, loss='exponential', max_depth=3,

                           max_features=None, max_leaf_nodes=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           min_samples_leaf=1, min_samples_split=2,

                           min_weight_fraction_leaf=0.0, n_estimators=100,

                           n_iter_no_change=None, presort='auto',

                           random_state=None, subsample=1.0, tol=0.0001,

                           validation_fraction=0.1, verbose=0,

                           warm_start=False)



from sklearn.svm import SVC

clf2 = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',

    kernel='linear', max_iter=-1, probability=True, random_state=230,

    shrinking=True, tol=0.001, verbose=False)



from sklearn.ensemble import RandomForestClassifier

clf3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

                       max_depth=2, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=10000,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)



from sklearn.naive_bayes import BernoulliNB

clf4 = BernoulliNB()



from sklearn.linear_model import LogisticRegression

clf5 = LogisticRegression(solver='liblinear')



from sklearn.tree import DecisionTreeClassifier

clf6 = DecisionTreeClassifier(criterion='entropy', max_depth=4)



from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('GradientBoosting', clf1),

                                    ('SVM', clf2),

                                    ('RandomForest', clf3),

                                    ('Bernouli', clf4),

                                    ('LogisticRegression', clf5),

                                    ('DecisionTree', clf6)],

                                    voting='hard')
from sklearn.model_selection import cross_val_score

scores = cross_val_score(eclf, df_train.drop(['Survived', 'PassengerId', 'Ticket'], axis=1), df_train['Survived'], cv=5)

print("Ensemble model")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
eclf.fit(df_train.drop(['Survived', 'PassengerId', 'Ticket'], axis=1), df_train['Survived'])

subm = eclf.predict(df_test.drop(['PassengerId', 'Ticket'], axis=1))



my_submission = pd.DataFrame({"PassengerId": df_test['PassengerId'], "Survived": subm})

my_submission.to_csv("prediction.csv", index=False)