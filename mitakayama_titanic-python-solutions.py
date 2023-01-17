import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
print('_'*40)
test.info()
fig, ax = plt.subplots()
sns.countplot('Survived', data=train)
ax.set_title('Survived')

plt.show()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('Pclass', hue='Survived', data=train)
ax.set_title('Pclass:Survived vs Dead')
plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('Sex', hue='Survived', data=train)
ax.set_title('Sex:Survived vs Dead')
plt.show()
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
f, ax = plt.subplots(1, 2, figsize=(18 ,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))
sns.violinplot("Sex","Age", hue="Survived", data=train, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('SibSp', hue='Survived', data=train)
ax.set_title('SibSp:Survived vs Dead')
plt.show()
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('Parch', hue='Survived', data=train)
ax.set_title('Parch:Survived vs Dead')
plt.show()
def getFamilySize(row):
    return row["SibSp"] + row["Parch"] + 1

def getIsAlone(row):
    if row['FamilySize'] == 1:
        return 1
    return 0

def fam_size(train, test):
    for i in [train, test]:
        i['FamilySize'] = i.apply(getFamilySize, axis=1)
        i['IsAlone'] = i.apply(getIsAlone, axis=1)
        
        i['FamCate'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test

train, test = fam_size(train, test)
train.head()
train[["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('FamilySize', hue='Survived', data=train)
ax.set_title('FamilySize:Survived vs Dead')
plt.show()
train[["IsAlone", "Survived"]].groupby(['IsAlone'], as_index=False)['Survived'].agg({'Survived': ['mean','count']})
f, ax = plt.subplots()
sns.countplot('IsAlone', hue='Survived', data=train)
ax.set_title('IsAlone:Survived vs Dead')
plt.show()
def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test


train, test = names(train, test)
pd.crosstab(train['Name_Title'], train['Sex'])
train[["Name_Title", "Age"]].groupby(['Name_Title'], as_index=False).mean()
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test

train, test = age_impute(train, test)

train.info()
print('_'*40)
test.info()
def cabin(train, test):
    for i in [train, test]:
        i['ExistCabin'] = i.apply(getCabinKind, axis=1)
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test

def getCabinKind(row):
    cabin = row['Cabin']
    if cabin == cabin:
        return 1
    return 0
    
train, test = cabin_num(train, test)
train, test = cabin(train, test)

f, ax = plt.subplots()
sns.countplot('ExistCabin', hue='Survived', data=train)
ax.set_title('ExistCabin:Survived vs Dead')
plt.show()
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test

train, test = embarked_impute(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train.info()
print('_'*40)
test.info()
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test

train, test = ticket_grouped(train, test)
train.head()
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


def drop(train, test, bye = ['PassengerId', 'FamilySize', 'IsAlone', 'ExistCabin']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'FamCate'])
train, test = drop(train, test)

train.info()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print("%.4f" % rf.oob_score_)


pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
predictions = rf.predict(test)

test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submit.csv', index=False)