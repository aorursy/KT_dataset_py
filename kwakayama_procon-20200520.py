# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
train.info()

train.describe()

train.hist(figsize = (12, 12))
train.isnull().sum()
test.isnull().sum()
data = pd.concat([train, test], sort=False)

data.isnull().sum()
print(data['Age'] )
#年齢と生死の関係

split_data = []

for survived in [0,1]:

    split_data.append(data[data.Survived==survived])



temp = [i['Age'].dropna() for i in split_data]

plt.hist(temp, histtype='barstacked')
#Nameの敬称によって、年齢を推定できる？

#年齢が欠損している人について、その敬称ごとの人数を算出

num_mr = ((data['Age'].isnull()) & (data['Name'].str.contains('Mr. '))).sum()

num_ms = ((data['Age'].isnull()) & (data['Name'].str.contains('Ms. '))).sum()

num_mrs = ((data['Age'].isnull()) & (data['Name'].str.contains('Mrs. '))).sum()

num_miss = ((data['Age'].isnull()) & (data['Name'].str.contains('Miss. '))).sum()

num_master = ((data['Age'].isnull()) & (data['Name'].str.contains('Master. '))).sum()

num_dr = ((data['Age'].isnull()) & (data['Name'].str.contains('Dr. '))).sum()



print(num_mr)

print(num_ms)

print(num_mrs)

print(num_miss)

print(num_master)

print(num_dr)



#合計が、年齢の欠損地の総数と一致することを確認

print(num_mr + num_ms + num_mrs + num_miss + num_master + num_dr)
#敬称ごとの年齢の平均と標準偏差を算出

age_avg = data['Age'].mean()

age_std = data['Age'].std()



data_mr = data[data['Name'].str.contains('Mr. ')]

data_ms = data[data['Name'].str.contains('Ms. ')]

data_mrs = data[data['Name'].str.contains('Mrs. ')]

data_miss = data[data['Name'].str.contains('Miss. ')]

data_master = data[data['Name'].str.contains('Master. ')]

data_dr = data[data['Name'].str.contains('Dr. ')]



age_mr_avg = data_mr['Age'].mean()

age_mr_std = data_mr['Age'].std()

age_ms_avg = data_ms['Age'].mean()

age_ms_std = data_ms['Age'].std()

age_mrs_avg = data_mrs['Age'].mean()

age_mrs_std = data_mrs['Age'].std()

age_miss_avg = data_miss['Age'].mean()

age_miss_std = data_miss['Age'].std()

age_master_avg = data_master['Age'].mean()

age_master_std = data_master['Age'].std()

age_dr_avg = data_dr['Age'].mean()

age_dr_std = data_dr['Age'].std()



print(age_mr_avg, age_mr_std)

print(age_ms_avg, age_ms_std)

print(age_mrs_avg, age_mrs_std)

print(age_miss_avg, age_miss_std)

print(age_master_avg, age_master_std)

print(age_dr_avg, age_dr_std)



print(age_avg)

print(age_mr_avg)

print(age_ms_avg)

print(age_mrs_avg)

print(age_miss_avg)

print(age_master_avg)

print(age_dr_avg)
#age_mrs_stdがnanになっているのは、要素が一つしかないから？

#敬称がmrsの人の年齢の欠損値は、age_mrs_avgを定数で使用することとする。

data_ms
#年齢の欠損値の補完

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Mr.')), np.random.randint(age_mr_avg - age_mr_std, age_mr_avg + age_mr_std), inplace=True)

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Ms.')), age_ms_avg, inplace=True)

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Mrs.')), np.random.randint(age_mrs_avg - age_mrs_std, age_mrs_avg + age_mrs_std), inplace=True)

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Miss.')), np.random.randint(age_miss_avg - age_miss_std, age_miss_avg + age_miss_std), inplace=True)

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Master.')), np.random.randint(age_master_avg - age_master_std, age_master_avg + age_master_std), inplace=True)

data['Age'].mask((data['Age'].isnull()) & (data['Name'].str.contains('Dr.')), np.random.randint(age_dr_avg - age_dr_std, age_dr_avg + age_dr_std), inplace=True)
#Embarked, Fareの欠損値の補完

data['Embarked'].fillna('S', inplace=True)

data['Fare'].fillna(data['Fare'].median(), inplace = True)



#データの変換

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data.isnull().sum()
split_data = []

for survived in [0,1]:

    split_data.append(data[data.Survived==survived])

    

temp = [i['SibSp'].dropna() for i in split_data]

plt.hist(temp, histtype='barstacked')
temp = [i['Parch'].dropna() for i in split_data]

plt.hist(temp, histtype='barstacked')
#使用しないデータ列の削除

#使用するのは Pclass, Sex, Age, Fare, Embarked, SibSp, Parch

delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



X_train = train.drop('Survived', axis=1)

Y_train = train['Survived']

X_test = test.drop('Survived', axis=1)
#ランダムフォレスト

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X_train, Y_train)



y_pred = clf.predict(X_test)
#csv出力

sub = gender_submission

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('titanic_procon_20200520.csv', index=False)