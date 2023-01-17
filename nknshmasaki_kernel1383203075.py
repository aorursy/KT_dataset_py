# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# 主催者側が用意してくれたデータを確認

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 学習用データ(train) と テスト用データ(test) を pandas で読み込み

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()
test.head()
test_shape = test.shape

train_shape = train.shape

 

print(test_shape)

print(train_shape)
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("学習データにおける女性の生存率:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("学習データにおける男性の生存率:", rate_men)
# a = pd.get_dummies(test.Sex).female

# a.head()

my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pd.get_dummies(test.Sex).female})

my_submission.head()
# you could use any filename. We choose submission here

# my_submission.to_csv('submission.csv', index=False)
# scikit-learn から ランダムフォレストの分類器を import

from sklearn.ensemble import RandomForestClassifier
# 学習データの整形( X から y を予測する、という形に整形する )

y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])



X.head()
# 適当にハイパーパラメータを設定して学習

model_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model_forest.fit(X, y)
# テストデータから予測

X_test = pd.get_dummies(test[features])

predictions = model_forest.predict(X_test)



# 結果が numpy 形式で得られることを確認

print(predictions)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
# train.describe()
# test.describe()
# # trainデータ内で欠損している値(null)を数えてみる

# train_null_val = train.isnull().sum()

# train_null_val.head(100)
# # nullの値を中央値で代替する

# train["Age"] = train["Age"].fillna(train["Age"].median())

# train_null_val = train.isnull().sum()

# train_null_val.head(100)