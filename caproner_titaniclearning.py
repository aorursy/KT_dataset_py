import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import plot_importance

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



train_x = train_data[['PassengerId']].copy()



train_y = train_data[['Survived']].copy()



train_data.head()
train_data.info()
train_data.describe()
train_data.describe(include='O')
train_data[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
train_x['Pclass'] = train_data['Pclass'].copy()

train_x.head()
name_split_data=train_data['Name'].str.split(',', expand=True)[1].str.split(expand=True)

name_split_data['Survived']=train_data['Survived']

name_split_data.head()
name_split_cnt = name_split_data.groupby(0, as_index=False)[0].agg({'cnt':'count'})

name_split_survived = name_split_data.groupby(0, as_index=False).mean()

pd.merge(name_split_cnt, name_split_survived, on=0).sort_values('cnt', ascending=False)
title_map = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4}

name_split_data[0] = name_split_data[0].map(title_map).copy()

name_split_data = name_split_data.fillna(value=5).copy()

name_split_data.groupby(0, as_index=False)[0].agg({'cnt':'count'})
train_x['Title'] = name_split_data[0].astype('int').copy()

train_x.head()
train_data[['Sex', 'Survived']].groupby('Sex', as_index=False).mean()
train_x['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0}).copy()

train_x.head()
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=50)
train_x['Age'] = train_data['Age'].copy()

train_x.head()
train_x['SibSp'] = train_data['SibSp'].copy()

train_x['Parch'] = train_data['Parch'].copy()

train_x['Fare'] = train_data['Fare'].copy()

train_x.head()
train_x['Embarked'] = train_data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3, np.NaN: 4}).astype('int').copy()

train_x.head()
pclass_one_hot=pd.get_dummies(train_x[['Pclass']].astype('str'))

pclass_one_hot['PassengerId']=train_x['PassengerId']

train_x=pd.merge(train_x, pclass_one_hot, on='PassengerId')

title_one_hot=pd.get_dummies(train_x[['Title']].astype('str'))

title_one_hot['PassengerId']=train_x['PassengerId']

train_x=pd.merge(train_x, title_one_hot, on='PassengerId')

embarked_one_hot=pd.get_dummies(train_x[['Embarked']].astype('str'))

embarked_one_hot['PassengerId']=train_x['PassengerId']

train_x=pd.merge(train_x, embarked_one_hot, on='PassengerId')

train_x=train_x.drop(['Pclass', 'Title', 'Embarked'], axis=1)

train_x
train_x=train_x.drop(['PassengerId'], axis=1)

train_x
def get_x(df):

    # 处理直接用的特征

    df_x = df[['PassengerId','Age','SibSp','Parch','Fare']].copy()

    # 处理Sex

    df_sex = pd.get_dummies(df[['Sex']])

    df_x['Sex'] = df_sex['Sex_male'].copy()

    # 处理Pclass

    df_pclass = pd.get_dummies(df[['Pclass']].astype('str'))

    df_pclass['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_pclass, on='PassengerId')

    # 处理Embarked

    df_embarked = pd.get_dummies(df[['Embarked']])

    df_embarked['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_embarked, on='PassengerId')

    # 处理Name

    name_split_data = df['Name'].str.split(',', expand=True)[1].str.split(expand=True)

    name_split_data[0] = name_split_data[0].map({"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4}).copy()

    name_split_data = name_split_data.fillna(value=5).rename(columns={0 : 'Title'}).copy()

    df_title = pd.get_dummies(name_split_data[['Title']].astype('str'))

    df_title['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_title, on='PassengerId')

    # 对缺失的数据补零

    df_x = df_x.fillna(value=0)

    # 去掉PassengerId并返回

    return df_x.drop(['PassengerId'], axis=1)



train_x = get_x(train_data)

train_y = train_data['Survived'].copy()



test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_x = get_x(test_data)

test_x
xgbooster = XGBClassifier()

score = cross_val_score(xgbooster, train_x, train_y, cv=5, scoring='accuracy')

xgbooster.fit(train_x, train_y)

pred_y = xgbooster.predict(test_x)

score
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": pred_y

    })

submission.to_csv('/kaggle/working/submission.csv', index=False)
test_data.info()
test_data.describe()
test_data['Name'].str.split(',', expand=True)[1].str.split(expand=True).groupby(0, as_index=False)[0].agg({'cnt':'count'})
train_data['AgeExist'] = (np.isnan(train_data['Age'])).map({False: 1, True: 0})

train_data.groupby('AgeExist', as_index=False)['AgeExist'].agg({'cnt': 'count'})
train_data['CabinNum'] = train_data['Cabin'].str.split().str.len().fillna(0).astype('int')

train_data.groupby('CabinNum', as_index=False)['CabinNum'].agg({'cnt': 'count'})
def get_x(df):

    # 处理直接用的特征

    df_x = df[['PassengerId','Age','SibSp','Parch','Fare']].copy()

    # 处理Sex

    df_sex = pd.get_dummies(df[['Sex']])

    df_x['Sex'] = df_sex['Sex_male'].copy()

    # 处理Pclass

    df_pclass = pd.get_dummies(df[['Pclass']].astype('str'))

    df_pclass['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_pclass, on='PassengerId')

    # 处理Embarked

    df_embarked = pd.get_dummies(df[['Embarked']])

    df_embarked['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_embarked, on='PassengerId')

    # 处理Name

    name_split_data = df['Name'].str.split(',', expand=True)[1].str.split(expand=True)

    name_split_data[0] = name_split_data[0].map({"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4}).copy()

    name_split_data = name_split_data.fillna(value=5).rename(columns={0 : 'Title'}).copy()

    df_title = pd.get_dummies(name_split_data[['Title']].astype('str'))

    df_title['PassengerId'] = df['PassengerId'].copy()

    df_x = pd.merge(df_x, df_title, on='PassengerId')

    # 对缺失的数据补零

    df_x = df_x.fillna(value=0)

    # 补上Age是否存在的特征

    df_x['AgeExist'] = (np.isnan(df['Age'])).map({False: 1, True: 0})

    # 补上Cabin数量的特征

    df_x['CabinNum'] = df['Cabin'].str.split().str.len().fillna(0).astype('int')

    # 去掉PassengerId并返回

    return df_x.drop(['PassengerId'], axis=1)



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_x = get_x(train_data)

train_y = train_data['Survived'].copy()



test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_x = get_x(test_data)

test_x
xgbooster = XGBClassifier()

score = cross_val_score(xgbooster, train_x, train_y, cv=5, scoring='accuracy')

xgbooster.fit(train_x, train_y)

pred_y = xgbooster.predict(test_x)

score
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": pred_y

    })

submission.to_csv('/kaggle/working/submission.csv', index=False)
train_data['Age'].fillna(train_data['Age'].dropna().median(), inplace=True)

train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_data['Fare'].fillna(train_data['Fare'].dropna().median(), inplace=True)

train_data['FareBand'] = pd.qcut(train_data['Fare'], 5)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
train_data.loc[train_data['Age'] <= 16.336, 'Age'] = 1

train_data.loc[(train_data['Age'] > 16.336) & (train_data['Age'] <= 32.252), 'Age'] = 2

train_data.loc[(train_data['Age'] > 32.252) & (train_data['Age'] <= 48.168), 'Age'] = 3

train_data.loc[(train_data['Age'] > 48.168) & (train_data['Age'] <= 64.084), 'Age'] = 4

train_data.loc[train_data['Age'] > 64.084, 'Age'] = 5

train_data['Age'] = train_data['Age'].astype('int')

train_data.loc[train_data['Fare'] <= 7.854, 'Fare'] = 1

train_data.loc[(train_data['Fare'] > 7.854) & (train_data['Fare'] <= 10.5), 'Fare'] = 2

train_data.loc[(train_data['Fare'] > 10.5) & (train_data['Fare'] <= 21.679), 'Fare'] = 3

train_data.loc[(train_data['Fare'] > 21.679) & (train_data['Fare'] <= 39.688), 'Fare'] = 4

train_data.loc[train_data['Fare'] > 39.688, 'Fare'] = 5

train_data['Fare'] = train_data['Fare'].astype('int')

train_data
def get_x(df):

    # 处理直接用的特征

    df_x = df[['PassengerId','SibSp','Parch', 'Age', 'Fare']].copy()

    # 处理Age

    df_x['Age'].fillna(df_x['Age'].dropna().median(), inplace=True)

    df_x.loc[df_x['Age'] <= 16.336, 'Age'] = 1

    df_x.loc[(df_x['Age'] > 16.336) & (df_x['Age'] <= 32.252), 'Age'] = 2

    df_x.loc[(df_x['Age'] > 32.252) & (df_x['Age'] <= 48.168), 'Age'] = 3

    df_x.loc[(df_x['Age'] > 48.168) & (df_x['Age'] <= 64.084), 'Age'] = 4

    df_x.loc[df_x['Age'] > 64.084, 'Age'] = 5

    df_x['Age'] = df_x['Age'].astype('int')

    # 处理Fare

    df_x['Fare'].fillna(df_x['Fare'].dropna().median(), inplace=True)

    df_x.loc[df_x['Fare'] <= 7.854, 'Fare'] = 1

    df_x.loc[(df_x['Fare'] > 7.854) & (df_x['Fare'] <= 10.5), 'Fare'] = 2

    df_x.loc[(df_x['Fare'] > 10.5) & (df_x['Fare'] <= 21.679), 'Fare'] = 3

    df_x.loc[(df_x['Fare'] > 21.679) & (df_x['Fare'] <= 39.688), 'Fare'] = 4

    df_x.loc[df_x['Fare'] > 39.688, 'Fare'] = 5

    df_x['Fare'] = df_x['Fare'].astype('int')

    # 处理Sex

    df_sex = pd.get_dummies(df[['Sex']])

    df_x['Sex'] = df_sex['Sex_male'].copy()

    # 处理Pclass

    df_x['Pclass'] = df['Pclass'].copy()

    # 处理Embarked

    df_x['Embarked'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3, np.NaN: 4}).astype('int').copy()

    # 处理Name

    name_split_data = df['Name'].str.split(',', expand=True)[1].str.split(expand=True)

    name_split_data[0] = name_split_data[0].map({"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4}).copy()

    name_split_data = name_split_data.fillna(value=5).rename(columns={0 : 'Title'}).copy()

    df_x['Title'] = name_split_data[['Title']].astype('int')

    # 对缺失的数据补零

    df_x = df_x.fillna(value=0)

    # 补上Age是否存在的特征

    df_x['AgeExist'] = (np.isnan(df['Age'])).map({False: 1, True: 0})

    # 补上Cabin数量的特征

    df_x['CabinNum'] = df['Cabin'].str.split().str.len().fillna(0).astype('int')

    # 去掉PassengerId并返回

    return df_x.drop(['PassengerId'], axis=1)



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_x = get_x(train_data)

train_y = train_data['Survived'].copy()



test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_x = get_x(test_data)

train_x
xgbooster = XGBClassifier()

score = cross_val_score(xgbooster, train_x, train_y, cv=5, scoring='accuracy')

xgbooster.fit(train_x, train_y)

pred_y = xgbooster.predict(test_x)

score.mean()
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": pred_y

    })

submission.to_csv('/kaggle/working/submission.csv', index=False)
plot_importance(xgbooster)

plt.show()