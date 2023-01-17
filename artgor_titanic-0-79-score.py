#Libraries

import pandas as pd

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import seaborn as sns

sns.set_style('whitegrid')

import re

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV
df_train = pd.read_csv('../input/train.csv', dtype={'Age': np.float64}, )



df_test = pd.read_csv('../input/test.csv', dtype={'Age': np.float64}, )
df_train.describe(include='all')
df_test.describe(include='all')
#Pclass. It seems that Pclass is useful and requires no changes.

df_train.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
df_train['Title'] = df_train['Name'].apply(lambda x: (re.search(' ([a-zA-Z]+)\.', x)).group(1))

df_test['Title'] = df_test['Name'].apply(lambda x: (re.search(' ([a-zA-Z]+)\.', x)).group(1))



df_train['Title'].value_counts(), df_test['Title'].value_counts()
#Dict

title_mapping = {

                    'Capt':       'Officer',

                    'Col':        'Officer',

                    'Major':      'Officer',

                    'Jonkheer':   'Royalty',

                    'Don':        'Royalty',

                    'Sir' :       'Royalty',

                    'Dr':         'Officer',

                    'Rev':        'Officer',

                    'Countess':   'Royalty',

                    'Dona':       'Royalty',

                    'Mme':        'Mrs',

                    'Mlle':       'Miss',

                    'Ms':         'Mrs',

                    'Mr' :        'Mr',

                    'Mrs' :       'Mrs',

                    'Miss' :      'Miss',

                    'Master' :    'Master',

                    'Lady' :      'Royalty'

                    } 

#Use dictionary to change values

for k,v in title_mapping.items():

    df_train.loc[df_train['Title'] == k, 'Title'] = v

    df_test.loc[df_test['Title'] == k, 'Title'] = v
#Age. Missing values for Age should be filled. I think that simple mean/median isn't good enough.

#After several tries I stopped at median by Title, Sex and Pclass.

#df_train.groupby(['Title']).mean()

#df_train.groupby(['Sex', 'Pclass', 'Title']).mean()

print(df_train.groupby(['Title', 'Sex', 'Pclass'])['Age'].median())
#Age. Fill NA with median by sex, pclass, title

df_train['Age'] = df_train.groupby(['Sex','Pclass','Title'])['Age'].apply(lambda x: x.fillna(x.median()))

df_test['Age'] = df_test.groupby(['Sex','Pclass','Title'])['Age'].apply(lambda x: x.fillna(x.median()))
df_train.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize=True)
#SibSp and Parch. These two variables allow to create a new variable for the size of the Family.

#At first I created a single feature showing whether the person had family. It wasn't good enough.

df_train['Family'] =  df_train['Parch'] + df_train['SibSp']

df_test['Family'] =  df_test['Parch'] + df_test['SibSp']
#Then I tried several variants and stopped on four groups: 0 relatives, 1-2, 3 and 5 or more.

# From the table we can see that such grouping makes sense

df_train.groupby(['Family']).mean()
#A function for Family transformation

def FamilySize(x):

    if x == 1 or x == 2:

        return 'little'

    elif x == 3:

        return 'medium'

    elif x >= 5:

        return 'big'

    else:

        return 'none'

#Applying it

df_train['Family'] = df_train['Family'].apply(lambda x : FamilySize(x))

df_test['Family'] = df_test['Family'].apply(lambda x : FamilySize(x))
#Just to see the survival rate.

#df_train.loc[df_train['Family'] == 'big']

df_train.groupby(['Pclass', 'Family'])['Survived'].mean()
#Ticket. We need to extract values from it. Function for extracting prefixes. Tickets have length of 1-3.



''' 

At first I also wanted to use Ticket numbers, but it was useless or prone to overfitting

def Ticket_Number(x):

    l=x.split()

    if len(x.split()) == 3:

        return x.split()[2]

    elif len(x.split()) == 2:

        return x.split()[1]

    else:

        return x.split()[0]

df_train['TicketNumber'] = df_train['Ticket'].apply(lambda x: Ticket_Number(x))        

df_test['TicketNumber'] = df_test['Ticket'].apply(lambda x: Ticket_Number(x))        

''' 



def Ticket_Prefix(x):

    l=x.split()

    if len(x.split()) == 3:

        return x.split()[0] + x.split()[1]

    elif len(x.split()) == 2:

        return x.split()[0]

    else:

        return 'None'

df_train['TicketPrefix'] = df_train['Ticket'].apply(lambda x: Ticket_Prefix(x))

df_test['TicketPrefix'] = df_test['Ticket'].apply(lambda x: Ticket_Prefix(x))
#Fare. There is only one missing value, and in test. Fill it with median for it Pclass.

ax = plt.subplot()

ax.set_ylabel('Average fare')

df_train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(7,3), ax = ax)

df_test['Fare'] = df_test.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))
#Cabin. I thought about ignoring this, but it turned out to be good.

#At first fill NA with 'Unknown',

df_train.Cabin.fillna('Unknown',inplace=True)

df_test.Cabin.fillna('Unknown',inplace=True)

#Extract first letter

df_train['Cabin'] = df_train['Cabin'].map(lambda x : x[0])

df_test['Cabin'] = df_test['Cabin'].map(lambda x : x[0])
#Now let's see. Most of the cabins aren't filled.

f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(y="Cabin", data=df_train, color="c")
#Other cabins vary.

sns.countplot(y="Cabin", data=df_train[df_train.Cabin != 'U'], color="c")
#Most of passangers with unknown Cabins died

sns.factorplot("Survived", col="Cabin",

               col_wrap=4, data=df_train[df_train.Cabin == 'U'],

               kind="count", size=2.5, aspect=.8)
#For passengers with known Cabins survival rate varies.

sns.factorplot("Survived", col="Cabin",

               col_wrap=4, data=df_train[df_train.Cabin != 'U'],

               kind="count", size=2.5, aspect=.8)
df_train.groupby(['Cabin']).mean()
#Embarked. Fill with most common value.

MedEmbarked = df_train.groupby('Embarked').count()['PassengerId']

df_train.Embarked.fillna(MedEmbarked,inplace=True)

df_test.Embarked.fillna(MedEmbarked,inplace=True)
#This is how the data looks like now.

df_train.head()
#For algorithms it is better to have dummies.

dummies = pd.get_dummies(df_train['Pclass'], prefix='Pclass')

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Pclass'], prefix='Pclass')

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['Title'])

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Title'])

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['Sex'])

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Sex'])

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['Cabin'], prefix='Cabin')

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Cabin'], prefix='Cabin')

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['Embarked'], prefix='Embarked')

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Embarked'], prefix='Embarked')

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['Family'], prefix='Family')

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['Family'], prefix='Family')

df_test = df_test.join(dummies)

dummies = pd.get_dummies(df_train['TicketPrefix'], prefix='TicketPrefix')

df_train = df_train.join(dummies)

dummies = pd.get_dummies(df_test['TicketPrefix'], prefix='TicketPrefix')

df_test = df_test.join(dummies)



#Drop unnecessary columns

to_drop = ['Pclass','Ticket', 'Name', 'SibSp', 'Sex', 'Parch', 'Cabin', 'Embarked', 'Title', 'Family', 'TicketPrefix']

for i in to_drop:

    df_train = df_train.drop([i], axis=1)

    df_test = df_test.drop([i], axis=1)
#This is how the data looks like now.

df_train.head()
#These variables will be used for learning

X_train = df_train.drop('Survived',axis=1)

Y_train = df_train['Survived']

X_test  = df_test.drop('PassengerId',axis=1).copy()
#Now to select features. This code ranks features by their importance fior Random Forest

clf = RandomForestClassifier(n_estimators=200)

clf = clf.fit(X_train, Y_train)

indices = np.argsort(clf.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking:')



for f in range(X_train.shape[1]):

    print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]],

                                      clf.feature_importances_[indices[f]]))
#This is automatical feature selection

model = SelectFromModel(clf, prefit=True)

train_new = model.transform(X_train)

train_new.shape
best_features=X_train.columns[indices[0:15]]

X = df_train[best_features]

Xt = df_test[best_features]

best_features
#Splitting data for tuning parameters for Random Forest.

X_train, X_test, y_train, y_test = train_test_split(X, Y_train, test_size=0.33, random_state=44)
#I saw this part of code there: https://www.kaggle.com/creepykoala/titanic/study-of-tree-and-forest-algorithms

#This is a great way to see how parameters influence the result of Random Forest

plt.figure(figsize=(9,7))



#N Estimators

plt.subplot(3,3,1)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(n_estimators=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.axis('tight')

plt.title('N Estimators')

plt.grid();



#Criterion

plt.subplot(3,3,2)

feature_param = ['gini','entropy']

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(criterion=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.title('Criterion')

plt.xticks(range(len(feature_param)), feature_param)

plt.grid();



#Max Features

plt.subplot(3,3,3)

feature_param = ['auto','sqrt','log2',None]

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_features=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(scores, '.-')

plt.axis('tight')

plt.title('Max Features')

plt.xticks(range(len(feature_param)), feature_param)

plt.grid();



#Max Depth

plt.subplot(3,3,4)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_depth=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Max Depth')

plt.grid();



#Min Samples Split

plt.subplot(3,3,5)

feature_param = range(2,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_samples_split =feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Min Samples Split')

plt.grid();



#Min Weight Fraction Leaf

plt.subplot(3,3,6)

feature_param = np.linspace(0,0.5,10)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Min Weight Fraction Leaf')

plt.grid();



#Max Leaf Nodes

plt.subplot(3,3,7)

feature_param = range(2,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_leaf_nodes=feature)

    clf.fit(X_train,y_train)

    scores.append(clf.score(X_test,y_test))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

plt.title('Max Leaf Nodes')

plt.grid();
forest = RandomForestClassifier(max_depth = 5,                                

                                min_samples_split =10,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 16)



parameter_grid = {'n_estimators' : [5, 8, 15],

                  'criterion' : ['gini', 'entropy'],

                  'max_features' : ['auto', 'sqrt', 'log2', None]

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(Y_train, n_folds=5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
forest = RandomForestClassifier(n_estimators = 8,

                                criterion = 'gini',

                                max_features = 'log2')

parameter_grid = {

                  'max_depth' : [None, 5, 10, 20],

                  'min_samples_split' : [5, 7],

                  'min_weight_fraction_leaf' : [0.0, 0.1, 0.2],

                  'max_leaf_nodes' : [4, 10, 16],

                 }



grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=StratifiedKFold(Y_train, n_folds=5))

grid_search.fit(X, Y_train)

print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
#These are good parameters

clf = RandomForestClassifier(n_estimators = 15,

                                criterion = 'gini',

                                max_features = 'sqrt',

                                max_depth = None,                                

                                min_samples_split =7,

                                min_weight_fraction_leaf = 0.0,

                                max_leaf_nodes = 18)



clf.fit(X, Y_train)

Y_pred_RF = clf.predict(Xt)

clf.score(X, Y_train)
submission = pd.DataFrame({

        'PassengerId': df_test['PassengerId'],

        'Survived': Y_pred_RF

    })

submission.to_csv('titanic.csv', index=False)