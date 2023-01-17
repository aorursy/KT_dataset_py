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
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

#from sklearn import grid_search

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import GridSearchCV



import seaborn as sb
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_submission = pd.read_csv('../input/gender_submission.csv')



#df1.to_csv('test.csv', index=False)
df_train.head() # 最初の５列を表示
df_train.describe() # 簡単な統計情報を表示
# 欠損値を表示

sb.heatmap(df_train.isnull(), cbar=False)
# 最初の５列を表示

df_test.head()
# 簡単な統計情報を表示

df_test.describe()
# 欠損値を表示

sb.heatmap(df_test.isnull(), cbar=False)
# 名前にMr.が付いている人物

df_mr = df_train[df_train.Name.str.contains('Mr.')]

df_mr = df_mr[df_mr['Age'].isnull() == False]

average_age_mr = df_mr['Age'].mean() #Mr.の乗客の平均年齢

print(average_age_mr)

print(df_mr['Age'].count())
# 名前にMiss.が付いている人物（未婚の女性）

df_miss = df_train[df_train.Name.str.contains('Miss.')]

df_miss = df_miss[df_miss['Age'].isnull() == False]

average_age_miss = df_miss['Age'].mean()

print(average_age_miss)

print(df_miss['Age'].count())
# 名前にMrs.が付いている人物（婚約済みの女性）

df_mrs = df_train[df_train.Name.str.contains('Mrs.')]

df_mrs = df_mrs[df_mrs['Age'].isnull() == False]

average_age_mrs = df_mrs['Age'].mean()

print(average_age_mrs)

print(df_mrs['Age'].count())
# 名前にMrs.が付いている人物（婚約済みの女性）

df_ms = df_train[df_train.Name.str.contains('Ms.')]

df_ms = df_ms[df_ms['Age'].isnull() == False]

average_age_ms = df_ms['Age'].mean()

print(average_age_ms)

print(df_ms['Age'].count())
# 名前にMaster.が付いている人物（男の子）

df_master = df_train[df_train.Name.str.contains('Master.')]

df_master = df_master[df_master['Age'].isnull() == False]

average_age_master = df_master['Age'].mean()

print(average_age_master)

print(df_master['Age'].count())
# 名前にDr.が付いている人物（医者？教授？）

df_dr = df_train[df_train.Name.str.contains('Dr.')]

df_dr = df_dr[df_dr['Age'].isnull() == False]

average_age_dr = df_dr['Age'].mean()

print(average_age_dr)

print(df_dr['Age'].count())
# 欠損した年齢を埋める（要修正）

x_train = df_train.copy()

x_train.loc[x_train['Name'].str.contains('Mr.') & x_train['Age'].isnull(), 'Age'] = average_age_mr

x_train.loc[x_train['Name'].str.contains('Miss.') & x_train['Age'].isnull(), 'Age'] = average_age_miss

x_train.loc[x_train['Name'].str.contains('Mrs.') & x_train['Age'].isnull(), 'Age'] = average_age_mrs

x_train.loc[x_train['Name'].str.contains('Ms.') & x_train['Age'].isnull(), 'Age'] = average_age_ms

x_train.loc[x_train['Name'].str.contains('Master.') & x_train['Age'].isnull(), 'Age'] = average_age_master

x_train.loc[x_train['Name'].str.contains('Dr.') & x_train['Age'].isnull(), 'Age'] = average_age_dr



t_train = x_train.copy()



sb.heatmap(x_train.isnull(), cbar=False) # 欠損値の視覚化

x_train.describe() # 統計情報

#x_train[x_train['Age'].isnull()] # NaNを表示
# 訓練データ



# 不要なデータを削除

x_train = x_train.drop(['PassengerId', 'Survived','Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)



#x_train = x_train.dropna()

#x_train = x_train.fillna(30) # 欠損値を埋める（特に年齢!）



x_train = x_train.replace({'male' : 1, 'female' : 0})



x_train['Family'] = x_train['SibSp'] + x_train['Parch'] + 1

#x_train = x_train.drop(['SibSp', 'Parch'], axis=1)



print(x_train.head())

print(x_train.shape)
# 欠損した年齢を埋める

x_test = df_test.copy()

x_test.loc[x_test['Name'].str.contains('Mr.') & x_test['Age'].isnull(), 'Age'] = average_age_mr

x_test.loc[x_test['Name'].str.contains('Miss.') & x_test['Age'].isnull(), 'Age'] = average_age_miss

x_test.loc[x_test['Name'].str.contains('Mrs.') & x_test['Age'].isnull(), 'Age'] = average_age_mrs

x_test.loc[x_test['Name'].str.contains('Ms.') & x_test['Age'].isnull(), 'Age'] = average_age_ms

x_test.loc[x_test['Name'].str.contains('Master.') & x_test['Age'].isnull(), 'Age'] = average_age_master

x_test.loc[x_test['Name'].str.contains('Dr.') & x_test['Age'].isnull(), 'Age'] = average_age_master



# Fareの欠損値を埋めておく（要検証）

x_test.loc[x_test['Fare'].isnull(), 'Fare'] = 35



sb.heatmap(x_test.isnull(), cbar=False) # 欠損値の視覚化

x_test.describe() # 統計情報

#x_test[x_test['Age'].isnull()] # NaNを表示

#x_test[x_test['Fare'].isnull()] # FareのNaNを表示
x_test.head()
# テストデータ

x_test = x_test.drop(['PassengerId','Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)



#x_test = x_test.dropna() # 欠損値を削除

#x_test = x_test.fillna(30) # 欠損値を埋める（特に年齢!）



x_test = x_test.replace({'male' : 1, 'female' : 0})



x_test['Family'] = x_test['SibSp'] + x_test['Parch'] + 1

#x_test = x_test.drop(['SibSp', 'Parch'], axis=1)



#print(x_test.head())

#print(x_test.shape)
# 訓練データ

#t_train = df_train.drop('Cabin', axis=1)

t_train = t_train.drop(['PassengerId','Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)



#t_train = t_train.dropna()

#t_train = t_train.fillna(30) # 欠損値を埋める（特に年齢!）



t_train = t_train['Survived']

print(t_train.head())

print(t_train.shape)
x_train = x_train.values

t_train = t_train.values

x_test = x_test.values

#print(data_x.shape)

#print(data_x)
sc = StandardScaler() # 標準化オブジェクト？

#sc.fit(x_train) # パラメータ計算

x_train_std = sc.fit_transform(x_train) # 訓練データを標準化

#sc.fit(x_test) # パラメータ計算

x_test_std = sc.fit_transform(x_test) # テストデータを標準化
#model = RandomForestClassifier(max_depth=5)

model = RandomForestClassifier(

    bootstrap=True,

    class_weight=None,

    criterion='gini',

    max_depth=10,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    min_samples_leaf=1,

    min_samples_split=15,

    min_weight_fraction_leaf=0.0,

    n_estimators=50,

    n_jobs=4,

    oob_score=False,

    random_state=0,

    verbose=0,

    warm_start=False)



model.fit(x_train_std, t_train) # 学習



pred_train = model.predict(x_train_std) # 予測を出力

accuracy_train = accuracy_score(t_train, pred_train) # 正答率を出力

print('訓練データでの正答率: %.2f' % accuracy_train)
pred_train = model.predict(x_test_std) # 予測を出力

df_answer = pd.DataFrame(pred_train, columns=['Survived'])

df_answer = df_answer.join(df_test['PassengerId'])

print(df_answer)

df_answer.to_csv('answer.csv', index=False)


from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV



param_grid = {

    'depth': [4,5,6,7,8,9,10]

}



fit_params = {

    'early_stopping_rounds': 100,

}



model = CatBoostClassifier(iterations=100)



gscv = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    #fit_params=fit_params,

    cv=3,

    n_jobs=-1,

    verbose=0,

    return_train_score=False

)

gscv.fit(x_train_std, t_train)

model = gscv.best_estimator_

pred_train = model.predict(x_train_std)

accuracy_score(t_train, pred_train)

print('訓練データでの正答率: %.2f' % accuracy_train)

pred_train = np.array(model.predict(x_test_std), dtype=int) # 予測を出力

df_answer = pd.DataFrame(pred_train, columns=['Survived'])

df_answer = df_answer.join(df_test['PassengerId'])

print(df_answer)

df_answer.to_csv('answer.csv', index=False)