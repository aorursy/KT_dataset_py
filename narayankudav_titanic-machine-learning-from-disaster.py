import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.columns
train.head()
train.tail()
train.shape
train.info()
males = train[train['Sex'] == 'male'][['Survived','PassengerId']]

males['Survived'] = males['Survived'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

males.groupby(['Survived']).count()['PassengerId'].plot(kind='pie', y='PassengerId', autopct='%1.0f%%', title='Male Passengers')
females = train[train['Sex'] == 'female'][['Survived','PassengerId']]

females['Survived'] = females['Survived'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

females.groupby(['Survived']).count()['PassengerId'].plot(kind='pie', autopct='%1.0f%%')
df1 = train[['Sex', 'Survived']]

df1['Survived'] = df1['Survived'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

sns.countplot(x='Sex', hue='Survived', data=df1)    
survived = train[train['Survived'] == 1][['Pclass','PassengerId']]

survived.groupby(['Pclass']).count()['PassengerId'].plot(kind='pie', y='PassengerId', autopct='%1.0f%%')
died = train[train['Survived'] == 0][['Pclass','PassengerId']]

died.groupby(['Pclass']).count()['PassengerId'].plot(kind='pie', y='PassengerId', autopct='%1.0f%%')
df7 = train[['Age', 'Pclass', 'Sex', 'Survived']]

df7['Survived'] = df7['Survived'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

sns.countplot(x='Pclass', hue='Survived', data=df7)                                     
df6 = train[['Age', 'Pclass', 'Sex', 'Survived']]

df6['Survived'] = df6['Survived'].apply(lambda x : '/Died' if (x == 0) else '/Survived')

df6['Category'] = df6['Sex']+df6['Survived']

sns.catplot(y='Age', x='Pclass', hue='Category', data=df6)
train['Age_Group'] = pd.cut(x=train['Age'], bins=8, labels=False, retbins=False, include_lowest=True)

age_group = train[['Age_Group', 'Survived']]

age_group.rename(columns={'Survived':'S'}, inplace=True)

age_group= age_group[pd.notnull(age_group['Age_Group'])]

age_group['Age_Group'] = (age_group['Age_Group'].apply(np.int64) +1 )*  10

age_group['S'] = age_group['S'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

survived= pd.get_dummies(age_group['S'], drop_first=False)

age_group = pd.concat([age_group, survived], axis=1)

del age_group['S']

dd = age_group.groupby(['Age_Group']).sum()

ax = dd.plot.bar(rot=0)
df = train[pd.notnull(train['Cabin'])]

df = df[['Cabin', 'Survived']]

df.rename(columns={'Survived':'S'}, inplace=True)

df['Cabin'] = df['Cabin'].str.strip()

df['Cabin'] = df['Cabin'].str.extract(r"([A-Z]).*")[0]

df['S'] = df['S'].apply(lambda x : 'Died' if (x == 0) else 'Survived')

ss = pd.get_dummies(df['S'], drop_first=False)

df = pd.concat([df, ss], axis=1)

del df['S']

df = df.groupby(['Cabin']).sum()

ax = df.plot.bar(rot=0)
train[pd.isnull(train['Age'])]
def impute_age(df):

    titleRegExp = r"(Mr|Mrs|Master|Miss|Rev|Sir|Dr|Ms|Lady|Col|Capt|Major|Mme|the Countess)[.|\s]"

    df = pd.concat([df, df['Name'].str.extract(titleRegExp)[[0]]], axis=1)

    df.rename(columns={0:'Title'}, inplace=True)

    df['Title'] = df.apply (lambda r: ('Miss' if (r['Title'] == 'Ms' ) else r['Title']), axis=1)

    df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'), inplace=True)

    del df['Title']

    return df
def impute_embarked(df):

    df['Embarked'].fillna('UNK',inplace=True)

    return df
def impute_cabin(df):

    df['Cabin'].fillna('UNK',inplace=True)

    return df
#sets the index to PassengerId. This field is not required for model

def set_index(df) :

    df.set_index(['PassengerId'], inplace=True)

    return df
def pre_processing(df):

    df = (df.pipe(impute_age)

            .pipe(impute_cabin)

            .pipe(impute_embarked)

            .pipe(set_index))

    return df
train = pre_processing(train)
train[pd.isnull(train['Age'])]
train[pd.isnull(train['Embarked'])]
train[pd.isnull(train['Cabin'])]
train.head()
train.tail()
def build_cabin_feature(df) :

    df['has_cabin'] = df.apply (lambda row: (0 if (row['Cabin'] == 'UNK') else 1), axis=1)

    df['Cabin'] = df['Cabin'].str.strip()

    df['Cabin'] = df['Cabin'].str.extract(r"([A-Z]).*")[0]

    cabin = pd.get_dummies(df['Cabin'], prefix='Cabin', drop_first=False)

    # drop column UNK which is related to NaN?

    del cabin['Cabin_U']

    df = pd.concat([df,cabin],axis=1)

    del df['Cabin']

    return df
def populate_sex(df):

    sex = pd.get_dummies(df['Sex'], drop_first=True)

    df = pd.concat([df,sex],axis=1)

    del df['Sex']

    return df
train['Age'].min(), train['Age'].max()
def populate_age(df):

    df['Age_Group'] = pd.cut(x=df['Age'], bins=8, labels=False, retbins=False, include_lowest=True)

    df['Age_G'] = df.apply (lambda r: ('Young' if (r['Age'] <= 20 ) else ('Adult' if (r['Age'] <= 40) else ( 'Old' if (r['Age'] <= 60) else 'Senior'))), axis=1)

    age = pd.get_dummies(df['Age_G'], drop_first=False)

    df = pd.concat([df,age],axis=1)

    del df['Age_G']

    del df['Age']

    return df
def populate_with_family(df):

    df['With_Family'] = df.apply (lambda row: (0 if (row['SibSp']+row['Parch'] == 0 ) else 1), axis=1)

    #del df['SibSp']

    #del df['Parch']

    return df
def populate_embarked(df):

    embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=False)

    if ('Embarked_UNK' in embarked.columns) :

        del embarked['Embarked_UNK']

    df = pd.concat([df,embarked],axis=1)

    del df['Embarked']

    return df
def delete_features(df, to_delete):

    df = df.drop(to_delete, axis=1)

    return df
def populate_features(df):

    df = (df.pipe(build_cabin_feature)

          .pipe(populate_sex)

          .pipe(populate_age)

          .pipe(populate_with_family)

          .pipe(populate_embarked)

          .pipe(delete_features, to_delete=['Name', 'Ticket', 'Fare']))  

    return df
train = populate_features(train)
train
sns.heatmap(train.corr(), annot=True).set_title("Corelation between features")

fig=plt.gcf()

fig.set_size_inches(20,20)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
train = delete_features(train, ['Cabin_T', 'has_cabin', 'Adult', 'Old', 'Senior', 'Young', 'Parch', 'With_Family'])
train
train.shape
y = train['Survived']

X = train

del X['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=100)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

def build_model(X, y):

    lr = LogisticRegression()

    rfe = RFE(lr, 10)             

    rfe = rfe.fit(X, y)

    

    X = X[X.columns[rfe.support_]]



    lr = LogisticRegression()

    rfe = RFE(lr, 10)             

    rfe = rfe.fit(X, y)



    return rfe, X
rfe, X_train = build_model(X_train, y_train)
X_test = X_test[X_train.columns]

y_pred = rfe.predict(X_test)

score = round(rfe.score(X_test,y_test) * 100,2)

print('Accuracy : {s:.2f}%'.format(s = score))
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='2.0f', cbar=False)

from sklearn.model_selection import GridSearchCV



param_grid = {'C': [0.001,0.01,0.1,10,100,1000]}

lr = LogisticRegression()

gm = GridSearchCV(lr, param_grid, cv=5)

gm_result = gm.fit(X_train, y_train)

best_score, best_params = gm_result.best_score_, gm_result.best_params_

print("Best: %f using %s" % (best_score, best_params))
y_pred = gm.predict(X_test)

score = round(gm.score(X_test,y_test) * 100,2)

print('Accuracy : {s:.2f}%'.format(s = score))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='2.0f', cbar=False)
X_sub  = pd.read_csv("/kaggle/input/titanic/test.csv")

X_passid = X_sub['PassengerId']

X_sub  = pre_processing(X_sub)

X_sub  = populate_features(X_sub)

X_sub  = X_sub[X_train.columns]

y_pred = gm.predict(X_sub)



y_pred = pd.DataFrame(y_pred.reshape(len(y_pred),1), columns=['Survived'])

res = pd.concat([X_passid, y_pred], axis=1)

res.to_csv('gender_submission.csv', index=False)
