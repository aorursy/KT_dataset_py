# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#データ読み込み、Shape

titanic_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

print("gender_submission  : " + str(titanic_gender.shape))



titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

print("train : " + str(titanic_train.shape))

titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")

print("test  : " + str(titanic_test.shape))
print(titanic_gender.dtypes)

titanic_gender.head()
print(titanic_train.dtypes)

titanic_train.head()
titanic_train.describe()
titanic_train.describe(exclude='number')
#欠損値

titanic_train.isnull().sum()
#男女別年齢ヒストグラム

import matplotlib.pyplot as plt



plt.figure(figsize=(15,3))



plt.subplot(1,3,1)

plt.title("Male")

plt.xlim([0,100])

plt.ylim([0,150])

plt.xlabel("Age")

plt.ylabel("Count")

plt.hist(x=titanic_train[titanic_train.Sex == 'male'].Age, bins=10)



plt.subplot(1,3,2)

plt.title("Female")

plt.xlim([0,100])

plt.ylim([0,150])

plt.xlabel("Age")

plt.ylabel("Count")

plt.hist(x=titanic_train[titanic_train.Sex == 'female'].Age, bins=10)



plt.subplot(1,3,3)

plt.pie(x=titanic_train.Sex.value_counts())

plt.legend(["male", "female"])



plt.show()
#前処理

#予測に使う項目

#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin']

#予測する項目

outputs = ['Survived']



#まとめて処理する

titanic_train['flag'] = 1

titanic_test['flag'] = 0

titanic_test['Survived'] = 0



titanic_all = pd.concat([titanic_train[features + outputs + ['flag']], titanic_test[features + outputs + ['flag']]])

print(titanic_all.shape)

titanic_all.head()
#cabinデータ

titanic_all[pd.isnull(titanic_all['Cabin']) == False]['Cabin'].head(30)
import math

#欠損値を平均値で埋める

titanic_all = titanic_all.fillna({'Age' : titanic_all['Age'].mean(), 'Fare' : titanic_all['Fare'].mean()})

#float > int

titanic_all['Age'] = titanic_all['Age'].apply(lambda x: math.ceil(x))

titanic_all['Fare'] = titanic_all['Fare'].apply(lambda x: math.ceil(x))



#Cabin処理(値が複雑なので、先頭1文字（Aなど）だけを使う）

titanic_all['Cabin_class'] = titanic_all['Cabin'].apply(lambda x: '' if pd.isnull(x) else x[0])

#titanic_all['Cabin_no'] = titanic_all['Cabin'].apply(lambda x: 0 if pd.isnull(x) else int(x[1:2]))

titanic_all.drop('Cabin', axis=1, inplace=True)



#category

titanic_all['Sex'] = titanic_all['Sex'].astype('category').cat.codes

titanic_all['Cabin_class'] = titanic_all['Cabin_class'].astype('category').cat.codes

print(titanic_all.dtypes)

titanic_all.head()
#features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin_class']

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin_class']

#features = ['Pclass', 'Sex', 'Age', 'Fare']



#trainとtestデータにわける

titanic_train_d = titanic_all[titanic_all.flag == 1]

titanic_train_d.drop('flag', axis=1, inplace=True)



titanic_test_d = titanic_all[titanic_all.flag == 0]

titanic_test_d.drop('flag', axis=1, inplace=True)

titanic_test_d.drop('Survived', axis=1, inplace=True)
#各項目の相関

print(titanic_train_d.corr())
#検証用にわける

from sklearn.model_selection import train_test_split

titanic_train_dx = titanic_train_d[features]

titanic_train_dy = titanic_train_d[outputs]

titanic_train_train_x, titanic_train_test_x, titanic_train_train_y, titanic_train_test_y = train_test_split(titanic_train_dx, titanic_train_dy, test_size=0.4)



print(titanic_train_train_x.shape)

print(titanic_train_train_y.shape)

print(titanic_train_test_x.shape)

print(titanic_train_test_y.shape)
#ランダムフォレスト

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



def RFC_Tuning(x, y, test_x, test_y, n, m, r):

    model_dt = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=r)

    model_dt.fit(x, y.values.ravel())

    preds_test = model_dt.predict(test_x)

    return (1.0 - mean_absolute_error(test_y, preds_test))



rfc_t = np.empty(4)

n_estimators_list = [50, 80, 100, 120]

max_depth_list = [3, 5, 8]

random_state_list = [0, 1, 2]

for max_d in max_depth_list:

    for n_est in n_estimators_list:

        for r_stat in random_state_list:

            rfc_t = np.vstack((rfc_t, [n_est, max_d, r_stat, RFC_Tuning(titanic_train_train_x, titanic_train_train_y, titanic_train_test_x, titanic_train_test_y, n_est, max_d, r_stat)]))



print("精度の高かった組み合わせ")

print(rfc_t[np.argmax(rfc_t[:, 3])])

n_est = int(rfc_t[np.argmax(rfc_t[:, 3])][0])

max_d = int(rfc_t[np.argmax(rfc_t[:, 3])][1])

r_stat = int(rfc_t[np.argmax(rfc_t[:, 3])][2])

        
from xgboost import XGBRegressor

model_xg = XGBRegressor()

model_xg.fit(titanic_train_train_x, titanic_train_train_y.values.ravel())

preds_test = model_xg.predict(titanic_train_test_x)

#結果がfloatになっているので

for i in range(len(preds_test)):

    preds_test[i] = int(np.round(preds_test[i]))

    

print(1.0 - mean_absolute_error(titanic_train_test_y, preds_test))

preds_test[0:10]
#提出用

model_dt = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=r_stat)

model_dt.fit(titanic_train_d[features], titanic_train_d[outputs].values.ravel())

preds_test = model_dt.predict(titanic_test_d)

output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': preds_test})



#model_xg = XGBRegressor()

#model_xg.fit(titanic_train_d[features], titanic_train_d[outputs].values.ravel())

#preds_test = model_xg.predict(titanic_test_d)

#preds_test_i = []

#for i in range(len(preds_test)):

#    preds_test_i.append(int(np.round(preds_test[i])))

#output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': preds_test_i})



output.to_csv('my_submission.csv', index=False)

print(output.shape)

output.head(10)
