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
# ##



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

%matplotlib inline



# ##



# 機械学習関連のライブラリ群



from sklearn.model_selection import train_test_split # 訓練データとテストデータに分割

# from sklearn.cross_validation import train_test_split # 訓練データとテストデータに分割

from sklearn.metrics import confusion_matrix # 混合行列



from sklearn.decomposition import PCA #主成分分析

from sklearn.linear_model import LogisticRegression # ロジスティック回帰

from sklearn.neighbors import KNeighborsClassifier # K近傍法

from sklearn.svm import SVC # サポートベクターマシン

from sklearn.tree import DecisionTreeClassifier # 決定木

from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト

from sklearn.ensemble import AdaBoostClassifier # AdaBoost

from sklearn.naive_bayes import GaussianNB # ナイーブ・ベイズ

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # 線形判別分析

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA # 二次判別分析

from lightgbm import LGBMClassifier # LightGBM
# ##



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# ##



train.head()



# ##



test.head()

# 形状把握

print("train_data_shape",train.shape)

print("test_data_shape",test.shape)

# ##



test_PassengerId = test.PassengerId



# ##



test_PassengerId.shape

# ##



test_PassengerId.head(10)

# ##



train.info()

# ##



test.info()

# ##



train.describe().T

# ##



test.describe().T

# train data

# df_train['Survived']=df_train['Survived'].astype('object')

train['Pclass']=train['Pclass'].astype('object')

train['SibSp']=train['SibSp'].astype('object')

train['Parch']=train['Parch'].astype('object')
# test data

test['Pclass']=test['Pclass'].astype('object')

test['SibSp']=test['SibSp'].astype('object')

test['Parch']=test['Parch'].astype('object')
train.info()
test.info()
# ヒストグラム

plt.figure(figsize=[7,7])

plt.hist(train.Pclass, bins=3, color='r', label='train', alpha=0.5, density=True)

plt.hist(test.Pclass, bins=3, color='b', label='test', alpha=0.5, density=True)

plt.legend()

plt.show()
sns.countplot(x='Survived',hue='Pclass',data=train)
# Name

# カウント / ユニーク数確認

train.Name.value_counts(sort=False)
# ヒストグラム

plt.figure(figsize=[7,7])

plt.hist(train.Sex, bins=3, color='r', label='train', alpha=0.5, density=True)

plt.hist(test.Sex, bins=3, color='b', label='test', alpha=0.5, density=True)

plt.legend()

plt.show()
sns.countplot(x='Survived',hue='Sex',data=train)
sns.distplot(train['Age'])
# ヒストグラム

plt.figure(figsize=[7,7])

plt.hist(train.SibSp, bins=9, color='r', label='train', alpha=0.5, density=True)

plt.hist(test.SibSp, bins=9, color='b', label='test', alpha=0.5, density=True)

plt.legend()

plt.show()
sns.countplot(x='Survived',hue='SibSp',data=train)
# ヒストグラム

plt.figure(figsize=[7,7])

plt.hist(train.Parch, bins=10, color='r', label='train', alpha=0.5, density=True)

plt.hist(test.Parch, bins=10, color='b', label='test', alpha=0.5, density=True)

plt.legend()

plt.show()
sns.countplot(x='Survived',hue='Parch',data=train)
# ヒストグラム

# Ticket

# カウント

train.Ticket.value_counts(sort=False)
# ヒストグラム

plt.figure(figsize=[21,7])

plt.hist(train.Fare, bins=30, color='r', label='train', alpha=0.5, density=True)

plt.hist(test.Fare, bins=30, color='b', label='test', alpha=0.5, density=True)

plt.legend()

plt.show()
sns.distplot(train['Fare'])
# ヒストグラム

# Cabin

# カウント

train.Cabin.value_counts(sort=True)
# Embarked

# ヒストグラム

# Embarked

# カウント

train.Embarked.value_counts(sort=True)
print(train.query('Embarked == "S"'))

print(train.query('Embarked == "C"'))

print(train.query('Embarked == "Q"'))
print(train.query('Embarked != "S" and Embarked != "C" and Embarked != "Q"'))
print(train.query('Fare >= 75.0 and Fare <= 85.0'))
print(train.query('Cabin == "B28"'))
print(train.query('Ticket == "113572"'))
# Embarked == "NaN" => Set "S" to Embarked

train.at[62, 'Embarked'] = 'S'

train.at[830, 'Embarked'] = 'S'
print(train.query('Ticket == "113572"'))
train.info()
train = train.drop('PassengerId', axis=1)

train = train.drop('Cabin', axis=1)
train.info()
test = test.drop('PassengerId', axis=1)

test = test.drop('Cabin', axis=1)
test.info()
train = train.fillna({'Age': train.Age.mean(), 'Fare': train.Fare.mean()})

test = test.fillna({'Age': test.Age.mean(), 'Fare': test.Fare.mean()})
print(train.info())

print(test.info())
train = train.drop('Name', axis=1)

train = train.drop('Ticket', axis=1)

test = test.drop('Name', axis=1)

test = test.drop('Ticket', axis=1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.info())

print(test.info())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform(train) 

train.loc[:,:] = scaled_values
train.corr()
color = sns.color_palette()



plt.figure(figsize=(30,24))

sns.heatmap(train.corr(),linecolor='white',linewidths=2,cmap='magma',square=True,annot=True)
train = train.drop('Sex_male', axis=1)

test = test.drop('Sex_male', axis=1)



train = train.drop('SibSp_0', axis=1)

test = test.drop('SibSp_0', axis=1)



train = train.drop('Parch_0', axis=1)

test = test.drop('Parch_0', axis=1)



train = train.drop('Embarked_S', axis=1)

test = test.drop('Embarked_S', axis=1)



train = train.drop('Pclass_3', axis=1)

test = test.drop('Pclass_3', axis=1)
plt.figure(figsize=(30,24))

sns.heatmap(train.corr(),linecolor='white',linewidths=2,cmap='magma',square=True,annot=True)
print(train.head(1))

print(train.shape)
print(test.head(1))

print(test.shape)
train['Parch_9'] = 0
print(train.head(1))

print(train.shape)
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.20, 

                                                    random_state=42)
X_train.shape
y_train.shape
X_test.shape
# 1st モデル LogisticRegression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
# Fitting the model on our trained dataset.

LR.fit(X_train,y_train)
print("Accuracy:",round(LR.score(X_train, y_train)*100,2))
LR.coef_
pred = LR.predict(test)
KNC = KNeighborsClassifier()
KNC.fit(X_train,y_train)
print("Accuracy:",round(KNC.score(X_train, y_train)*100,2))
pred = KNC.predict(test)
LGBM = LGBMClassifier()

LGBM.fit(X_train,y_train)

print("Accuracy:",round(LGBM.score(X_train, y_train)*100,2))
pred = LGBM.predict(test)
print(pred)
columns = ['Survived']

output = pd.DataFrame(pred, columns=columns, dtype='int64')

output
output = pd.concat([test_PassengerId, output], axis=1)

output
output.to_csv('submission.csv',index=False)