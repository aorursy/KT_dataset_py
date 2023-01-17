# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns 

import sklearn as sk

import matplotlib as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train['Age'].isnull().value_counts()
train.loc[train.Age.isnull(),'Age']=train[~train.Age.isnull()].Age.mean()
train['Age'].isnull().value_counts()
train.loc[train.Cabin.isnull(),'Cabin']='No Cabin'
train.info()
print(train.Embarked.value_counts())

train.loc[train.Embarked.isnull(),'Embarked']='S' #southampton
train.isnull().sum()
fig , axes = plt.pyplot.subplots(1,2,figsize=(25,8))

print(axes)

print(fig)
fig,axes = plt.pyplot.subplots(1, 2,figsize=(10,4))

print(axes)



sns.boxplot(x='Pclass',y='Age',data=train, palette='viridis',ax=axes[0])



# We now need to check for outliers (values that seem irregular compared to the others)

sns.boxplot(x='Pclass',y='Fare',data=train, palette='viridis',ax=axes[1])



plt.pyplot.show()

train.loc[train.Fare>200]
numerical_column = ['int64','float64'] #select only numerical features to find correlation

plt.pyplot.figure(figsize=(10,10))

sns.heatmap(

    train.select_dtypes(include=numerical_column).corr(),

    cmap=plt.cm.RdBu,

    vmax=1.0,

    linewidths=0.1,

    linecolor='white',

    square=True,

    annot=True

)
train.Sex.value_counts()
from sklearn import preprocessing

train['Sex']=preprocessing.LabelEncoder().fit_transform(train['Sex'])

train.Sex.value_counts()

palette ={1:"g", 0:"r"}

sns.countplot(x='Sex',data=train,hue="Survived", palette=palette)
def features_engineering(df):

    df.loc[df.Age.isna(), 'Age'] = df[~df.Age.isna()].Age.mean()

    df.loc[df.Cabin.isna(),'Cabin'] = "No Cabin"

    df.loc[df.Embarked.isna(),'Embarked'] = "S"

    df['persons_abroad_size'] = (df['Parch']+df['SibSp']).astype(int)

    df['alone'] = np.where(df['Parch']==0,1,0)

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    df['Sex'] = df['Sex'].map( {'male': 1, 'female': 2} ).astype(int)

    df['log_fare'] = df['Fare'].apply(np.log)

    df['Room'] = (df['Cabin']

                    .str.slice(1,5).str.extract('([0-9]+)', expand=False)

                    .fillna(0)

                    .astype(int))

    df['RoomBand'] = 0

    df.loc[(df.Room > 0) & (df.Room <= 20), 'RoomBand'] = 1

    df.loc[(df.Room > 20) & (df.Room <= 40), 'RoomBand'] = 2

    df.loc[(df.Room > 40) & (df.Room <= 80), 'RoomBand'] = 3

    df.loc[df.Room > 80, 'RoomBand'] = 4

    df_id = df.PassengerId

    df = df.drop('PassengerId', axis=1)

    return df,df_id
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train,train_id = features_engineering(train)

test,test_id = features_engineering(test)
train.info()
import xgboost as xgb

from sklearn import model_selection

X_train = train.drop('Survived',axis=1).select_dtypes(include=['int32','int64','float64'])

y_train = train['Survived']

X_test = test.select_dtypes(include=['int32','int64','float64'])



xg_boost = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.65, gamma=2, learning_rate=0.3, max_delta_step=1,

       max_depth=4, min_child_weight=2, missing=None, n_estimators=280,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)
xg_boost.fit(X_train, y_train)
print(xg_boost.score(X_train, y_train))



scores = model_selection.cross_val_score(xg_boost, X_train, y_train, cv=5, scoring='accuracy')

print(scores)

print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
xgb.plot_importance(xg_boost)

plt.pyplot.show()
Y_pred = xg_boost.predict(X_test)
submission = pd.DataFrame({

    "PassengerId": test_id, 

    "Survived": Y_pred 

})

submission.head(10)
submission.to_csv('submission.csv', index=False)
submission.to_csv('deepansh.csv')