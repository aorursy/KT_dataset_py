import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plotting
import seaborn as sns # more plotting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_test = [train, test]
train.head()
test
for df in train_test:
    df['Sex'] = df['Sex'].replace({'male': '1', 'female': '0'})
    df['Sex'] = df['Sex'].astype(int)
    
    for col in df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']].columns:
        print("---- %s ---" % col)
        print(df[col].value_counts(dropna = False))

print(train['Survived'].value_counts(dropna = False))
for df in train_test:
    i = 0
    for e in df['Name']:
        df.loc[i, 'Title'] = e.split(', ')[1].split('.')[0]
        i += 1

print(train_test[0]['Title'])
for df in train_test:
    df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']].hist()
train['Survived'].hist()
for df in train_test:
    df[['Age', 'Fare']].hist()
    print(df[['Age', 'Fare']].info())
    print(df['Age'].describe())
    print(df['Fare'].describe())
for df in train_test:
    print(df.groupby('Embarked')['Fare'].mean())
train.loc[train['Embarked'].isnull()]
train_test[0] = train_test[0].loc[train_test[0]['Embarked'].isnull() == False]

for df in train_test:
    df.loc[df['Age'].isnull(),'hasAge'] = 0
    df['hasAge'] = df['hasAge'].fillna(1)
    
    df['Age'] = df['Age'].fillna(-1.0)
    df.loc[((df['Age']/0.5)%2 == 1) & (df['Age'] >= 1.0), 'hasAge'] = 0
    df.loc[((df['Age']/0.5)%2 == 0) & (df['Age'] >= 1.0), 'hasAge'] = 2
    df.loc[df['Age'] < 1.0, 'hasAge'] = 2
    df.loc[df['Age'] == -1.0, 'hasAge'] = 1
    
print(train_test[0]['hasAge'].value_counts())
print(train['Embarked'])
for df in train_test:
    df.loc[df['Cabin'].isnull(), 'hasCabin1'] = 0
    df['hasCabin1'] = df['hasCabin1'].fillna(1)
    df['Cabin'] = df['Cabin'].fillna("Z999")
    df['CabinL'] = df['Cabin'].astype(str).str[0]
    
    for i in range(1, 5, 1):
        df['CabinN%d' % (i)] = "999"

    df.loc[df['Cabin'].str.len() <= 4.0, 'CabinN1'] = df['Cabin'].astype(str).str[1:]
    df.loc[df['Cabin'].str.len() == 5.0, 'CabinN1'] = df['Cabin'].astype(str).str[3:]
    
    for i in [7, 11, 15]:
        for j in range(1, i-1, 4):
            df.loc[df['Cabin'].str.len() == i, 'CabinN%d' % ((j+3)/4)] = df['Cabin'].astype(str).str[j:j+2]

    df.loc[df['Cabin'].str.len() == 1, 'CabinN1'] = "999"
    
    for i in range(1, 5, 1):
        df['CabinN%d' % (i)] = pd.to_numeric(df['CabinN%d' % (i)], downcast='integer')

    df['CabinN1'] = df['CabinN1'].astype(int)
    
    for i in range(2, 5, 1):
        df.loc[df['CabinN%d' % (i)] == 999, 'hasCabin%d' % (i)] = 0
        df['hasCabin%d' % (i)] = df['hasCabin%d' % (i)].fillna(1)
        

print(train_test[0]['CabinL'].value_counts(dropna=False))
print(train_test[0].groupby('CabinL')['Survived'].mean())
test.loc[test['Fare'].isnull(), 'Fare'] = test['Fare'].mean()

for df in train_test:
    df['Fare'] = round(df['Fare'], 1)
f, ax = plt.subplots(figsize = (30, 30))

corrmat = train_test[0].corr()
sns.heatmap(corrmat, annot = True)
f, ax = plt.subplots(figsize = (30, 30))

corrmat = train_test[1].corr()
sns.heatmap(corrmat, annot = True)
print(train_test[0].groupby('hasAge')['Survived'].mean())
def combine_features(feature1, feature2):
    for df in train_test:
        for i in range(1, train_test[0][feature1].unique().size+1):
            for j in range(1, train_test[0][feature2].unique().size+1):        
                 df.loc[(df[feature1] == train_test[0][feature1].unique()[i-1]) \
                 &      (df[feature2] == train_test[0][feature2].unique()[j-1]), feature1 + '_' + feature2] \
                    = (i-1)*train_test[0][feature2].unique().size+j
    print(train_test[0].groupby(feature1+'_'+feature2)['Survived'].mean())
combine_features('Sex', 'Pclass')
for df in train_test:
    df['isAlone'] = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'isAlone'] = 1
combine_features('Pclass', 'isAlone')
combine_features('Sex', 'isAlone')
combine_features('Pclass', 'hasCabin1')
combine_features('Sex_isAlone', 'Pclass_hasCabin1')

for df in train_test:
    df = df.rename(columns = {'Sex_isAlone_Pclass_hasCabin1': 'SAPC'}, inplace = True)
for df in train_test:
    df['ageCat'] = pd.cut(df['Age'], [-2.0, -0.5, 14.0, 24.0, 40.0, 60.0, 100.0], labels = [1, 2, 3, 4, 5, 6])
for df in train_test:
    df['famSize'] = df['SibSp'] + df['Parch']

print(train_test[0].groupby('famSize')['Survived'].mean())
for df in train_test:
    for i in range(0, 3):
        df.loc[df['famSize'] >= sum(range(0, i+1)), 'famCat'] = i

print(train_test[0].groupby('famCat')['Survived'].mean())
for df in train_test:
    df['farePP'] = (df['Fare'] / (df['famSize'] + 1.0)).round(1)
    
print(train_test[0]['farePP'])
for df in train_test:
    df['fareCat'] = pd.cut(df['farePP'], [-1.0, 8.6, 26.0, 1000.0], labels = [1, 2, 3])

print(train_test[0].groupby('fareCat')['Survived'].mean())
combine_features('SAPC', 'fareCat')

for df in train_test:
    df = df.rename(columns = {'SAPC_fareCat': 'SAPCF'}, inplace = True)
for e in train_test[0].groupby('SAPCF'):
    print(e)
for df in train_test:
    for town in ['S', 'Q', 'C']:
        df.loc[df['Embarked'] == town, 'Embarked'] = ['S', 'Q', 'C'].index(town)

print(train_test[0].groupby('Embarked')['Survived'].mean())
for df in train_test:
    df.loc[(df['Title'] != 'Mr') & (df['Title'] != 'Mrs') & (df['Title'] != 'Miss') & (df['Title'] != 'Master'), 'Title'] = 'Other'
    for title in ['Mr', 'Mrs', 'Miss', 'Master', 'Other']:
        df.loc[df['Title'] == title, 'Title'] = ['Mr', 'Mrs', 'Miss', 'Master', 'Other'].index(title)
    df['Title'] = pd.to_numeric(df['Title'])


print(train_test[0]['Title'])
print(train_test[0].columns)

#classifier = DecisionTreeClassifier(max_depth = 10, random_state = 1)

#train_dummies = pd.get_dummies(pd.DataFrame())

splitter = train_test[0][['SAPCF', 'Survived', 'Title', 'Embarked', 'ageCat', 'hasAge']]

#res = cross_val_score(clf, X, y, scoring='accuracy', cv = 5)
train_set, fake_test = train_test_split(splitter, test_size = 0.2, random_state = 1)
train_set_true = train_set['Survived']
train_set = train_set.loc[:, train_set.columns != 'Survived']
fake_test_true = fake_test['Survived']
fake_test = fake_test.loc[:, train_set.columns != 'Survived']

splitter_true = splitter['Survived']
splitter_set = splitter.loc[:, splitter.columns != 'Survived']
#classifier.fit(train_set, train_set_true)

#print(classifier.feature_importances_)
models = [DecisionTreeClassifier(random_state = 1), RandomForestClassifier(random_state = 1), AdaBoostClassifier(random_state = 1), GradientBoostingClassifier(random_state = 1)]

for model in models:
    res = cross_val_score(model, splitter.loc[:, splitter.columns != 'Survived'], splitter['Survived'], scoring='accuracy', cv = 5)
    print(res)
#%time dectreeclass = GridSearchCV(DecisionTreeClassifier(random_state = 1), {'min_samples_split': [1.0, 2, 3], 'min_samples_leaf': [7, 8, 9], 'max_depth': range(1, 8, 1), 'splitter': ['best', 'random']}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(dectreeclass.best_estimator_)
#%time adaboostclass = GridSearchCV(AdaBoostClassifier(random_state = 1), {'n_estimators': range(25, 35, 1), 'learning_rate': [1.1, 1.2, 1.3]}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(adaboostclass.best_estimator_)
%time gradboostclass = GridSearchCV(GradientBoostingClassifier(random_state = 1), {'min_samples_split': range(2, 4, 1), 'min_samples_leaf': range(2, 4, 1), 'learning_rate': [0.4, 0.5, 0.6], 'n_estimators': range(1, 10, 1), 'max_depth': [2, 3, 4]}, cv = 3, refit = True, scoring = 'roc_auc').fit(splitter_set, splitter_true)

print(gradboostclass.best_estimator_)
#%time randforclass = GridSearchCV(RandomForestClassifier(random_state = 1), {'n_estimators': range(50, 60, 1), 'max_depth': [4, 5, 6]}, cv = 3, refit = True, scoring = 'roc_auc').fit(splitter_set, splitter_true)

#print(randforclass.best_estimator_)
#%time logreg = GridSearchCV(LogisticRegression(random_state = 1), {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'tol': [0.01, 0.1, 1, 10, 100]}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(logreg.best_estimator_)
#classifiers = [dectreeclass, adaboostclass, gradboostclass, randforclass, logreg]

#for classifier in classifiers:
#    fake_test_pred = classifier.predict(fake_test)
#    print(accuracy_score(fake_test_true, fake_test_pred))
fake_test_pred = gradboostclass.predict(fake_test)
print(accuracy_score(fake_test_true, fake_test_pred))
tester = train_test[1][['SAPCF', 'Title', 'Embarked', 'ageCat', 'hasAge']]

tester
#tester = tester.fillna(999)
#test['CabinL_T'] = 0

pred_df = pd.DataFrame({'PassengerId': range(892,1310),'Survived': gradboostclass.predict(tester)})

pred_df.to_csv('submission.csv', index=False)
pred_df