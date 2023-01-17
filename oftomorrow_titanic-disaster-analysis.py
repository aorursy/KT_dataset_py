import pandas as pd

import numpy as np



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='whitegrid')

%matplotlib inline



#scikit-learn

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

from sklearn import svm



#read the data

train = pd.read_csv('../input/train.csv', index_col='PassengerId')

test = pd.read_csv('../input/test.csv', index_col='PassengerId')



train.info()
test.info()
train.describe()
#fill one missed value with mean

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

#take a look at the embarked

sns.countplot(x='Embarked', data=train);
#fill an empty values with the most common

train['Embarked'] = train['Embarked'].fillna('S')
def fill_age(data):

    max_data = data['Age'].mean() + data['Age'].std()

    min_data = data['Age'].mean() - data['Age'].std()

    rand_data = np.random.randint(min_data, max_data, size=data['Age'].isnull().count())

    data.loc[np.isnan(data['Age']), ['Age']] = rand_data

    return data





fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(9, 4))

axis1.set_title('Original')

axis2.set_title('Filled')

train['Age'].dropna().hist(bins=70, ax=axis1);



train = fill_age(train)

test = fill_age(test)



train['Age'].hist(bins=70, ax=axis2);
g1 = sns.factorplot('Survived', col='Pclass', col_wrap=4, data=train, 

                    kind="count", order=[1,0], aspect=0.6)

(g1.set_xticklabels(['Yes', 'No'])

 .set_titles('{col_name} Class'));

g2 = sns.factorplot('Survived', col='Sex', col_wrap=4, data=train, 

                    kind="count", order=[1,0], aspect=0.9)

(g2.set_xticklabels(['Yes', 'No'])

 .set_titles('{col_name}'));

g3 = sns.factorplot(x='Sex', y='Survived', col='Pclass', data=train, 

                    kind='bar', ci=None, aspect=0.6)

(g3.set_axis_labels('', 'Survival Rate')

 .set_xticklabels(['Men', 'Women'])

 .set_titles('{col_name} Class')

 .set(ylim=(0, 1)));
g4 = sns.violinplot(x='Survived', y='Age', hue='Sex', data=train, split=True, order=[1,0])

(g4.set_xticklabels(['Yes', 'No']));
def f(Age):

    for i in range(10, int(train['Age'].max()+1), 10):

        if Age >= (i-10) and Age <= i:

            return str(i-10) + '-' + str(i)



group_data = train[train['Survived'] == 0].loc[:,['Age', 'Pclass', 'Survived']]

group_data['AgeRange'] = group_data['Age'].apply(f)

del(group_data['Age'])

group_data = group_data.groupby(['AgeRange', 'Pclass']).count().reset_index()



pivoted = group_data.pivot(index='AgeRange', columns='Pclass', values='Survived').fillna(0)

sns.heatmap(pivoted);
f, ax = plt.subplots()



sunk = train.loc[train['Survived']==0, ['Age']]

sns.distplot(sunk,

            label='Sunk', bins=80, kde=False, color='r')



survived = train.loc[train['Survived']==1, ['Age']]

sns.distplot(survived,

            label='Survived', bins=80, kde=False, color='b')



ax.legend(ncol=2, loc='upper right', frameon=True)

ax.set(xlabel='Ages');
from scipy.stats import spearmanr



def demographic_category(p):

    age, sex = p

    if age < 18:

        return 'Child'

    elif age > 65:

        return 'Elderly'

    else:

        return sex



train['Demographic'] = train[['Age', 'Sex']].apply(demographic_category, axis=1)

test['Demographic'] = test[['Age', 'Sex']].apply(demographic_category, axis=1)



g5 = sns.countplot(x='Survived', hue='Demographic', data=train, order=[1,0], palette='Set2')

(g5.set_xticklabels(['Yes', 'No']));
g6 = sns.factorplot('Survived', col='Demographic', 

                    data=train, kind='count', palette='Set2', order=[1,0], size=3, aspect=0.6)

(g6.set_xticklabels(['Yes', 'No'])

 .set_titles('{col_name}'));
g7 = sns.factorplot(x='Demographic', y='Survived', col='Pclass', 

                    data=train, kind='bar', palette='Set2', ci=None, aspect=0.6)

(g7.set_axis_labels('', 'Survival Rate')

 .set_titles('{col_name} Class')

 .set(ylim=(0, 1)));
sns.factorplot('Demographic', col='Embarked', 

                    data=train, kind='count', palette='Set2', ci=None, aspect=0.6);
g8 = sns.countplot(x='Survived', hue='Parch', data=train, order=[1,0], palette='Set2')

(g8.set_xticklabels(['Yes', 'No']));
g7 = sns.countplot(x='Survived', hue='SibSp', data=train, order=[1,0], palette='Set2')

(g7.set_xticklabels(['Yes', 'No']));
g8 = sns.countplot(x='Survived', hue='Embarked', data=train, order=[1,0], palette='Set2')

(g8.set_xticklabels(['Yes', 'No']));
f, ax = plt.subplots()



sunk = train.loc[train['Survived']==0, ['Fare']]

sns.distplot(sunk,

            label='Sunk', bins=50, kde=False, color='r')



survived = train.loc[train['Survived']==1, ['Fare']]

sns.distplot(survived,

            label='Survived', bins=50, kde=False, color='b')



ax.legend(ncol=2, loc='upper right', frameon=True)

ax.set(xlabel='Fares');
embarked = pd.get_dummies(train['Embarked'])

embarked.columns = ['S', 'C', 'Q']

demographic = pd.get_dummies(train['Demographic'])

demographic.columns = ['Child', 'Elderly', 'Female', 'Male']

train = train.join(embarked)

train = train.join(demographic)



embarked = pd.get_dummies(test['Embarked'])

embarked.columns = ['S', 'C', 'Q']

demographic = pd.get_dummies(test['Demographic'])

demographic.columns = ['Child', 'Elderly', 'Female', 'Male']

test = test.join(embarked)

test = test.join(demographic)



def family(p):

    parch, sibsp = p

    if (parch + sibsp) > 0:

        return 1

    else:

        return 0



train['Family'] = train[['Parch', 'SibSp']].apply(family, axis=1)

test['Family'] = test[['Parch', 'SibSp']].apply(family, axis=1)



y_train = train.loc[:, ['Survived']]

X_train = train.drop(['Sex', 'Ticket', 'Cabin', 'Name', 'Demographic', 

                      'Embarked', 'Parch', 'SibSp', 'Male', 'S', 'Survived'], axis=1)

X_test = test.drop(['Sex', 'Ticket', 'Cabin', 'Name', 'Demographic', 

                    'Embarked', 'Parch', 'SibSp', 'Male', 'S'], axis=1)



g = sns.countplot(x='Survived', hue='Family', data=train, order=[1,0], palette='Set2')

(g.set_xticklabels(['Yes', 'No']));
regression = LogisticRegression()

regression.fit(X_train, np.ravel(y_train))

lr_score = regression.score(X_train, y_train)



tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

prediction = tree.predict(X_test)

tree_score = tree.score(X_train,y_train)



knc = KNeighborsClassifier(n_neighbors=25)

knc.fit(X_train, np.ravel(y_train))

knc_score = knc.score(X_train, y_train)



svc = svm.SVC()

svc.fit(X_train, np.ravel(y_train))

svc_score = svc.score(X_train,y_train)



#write prediction to file

submission = pd.DataFrame({

        'PassengerId': test.index,

        'Survived': prediction

    })

submission.to_csv('titanic.csv', index=False)



#look at the scores

#I choose KNC

print("LogisticRegression score:\t\t\t%0.5f" % lr_score)

print("DecisionTreeClassifier score:\t\t\t%0.5f" % tree_score)

print("KNeighborsClassifier score:\t\t\t%0.5f" % knc_score)

print("SVC score:\t\t\t\t\t%0.5f\n" % svc_score)
importance = pd.DataFrame(X_train.columns)

importance.columns = ['Features']

importance["Importance"] = pd.Series(tree.feature_importances_)

importance