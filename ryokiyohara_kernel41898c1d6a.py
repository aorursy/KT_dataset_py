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


1

2

3

4

5

6

7

8

import pandas as pd

import numpy as np

 

train = pd.read_csv("../ディレクトリを指定/train.csv")

test = pd.read_csv("../ディレクトリを指定/test.csv")

 

1

2

3

4

5

6

7

8

test_shape = test.shape

train_shape = train.shape

 

print(test_shape)

print(train_shape)

 

 

1

2

3

4

5

(418, 11)

(891, 12)

 

 

 

1

2

3

4

5

test.describe()

train.describe()

 

 

1

2

3

4

5

6

7

8

9

10

11

12

13

def kesson_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        kesson_table = pd.concat([null_val, percent], axis=1)

        kesson_table_ren_columns = kesson_table.rename(

        columns = {0 : '欠損数', 1 : '%'})

        return kesson_table_ren_columns

 

kesson_table(train)

kesson_table(test)

 

 

1

2

3

4

5

6

7

8

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")

 

 

kesson_table(train)

 

 

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

train.head(10)



1

2

3

4

5

6

7

8

9

10

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

 

train.head(10)

 

 

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()

test.head(10)



1

2

3

4

5

6

7

8

9

10

11

12

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

test.Fare[152] = test.Fare.median()

 

test.head(10)

 

 

1

2

3

4

5

# scikit-learnのインポートをします

from sklearn import tree

 

 



1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

# 「train」の目的変数と説明変数の値を取得

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数の値を取得

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)

 

  

1

2

3

4

5

# 予測データのサイズを確認

my_prediction.shape

 

(418,)



1

2

3

4

(418,)

 

 

  

#予測データの中身を確認

print(my_prediction)



1

2

3

4

5

#予測データの中身を確認

print(my_prediction)

 

 

# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])



1

2

3

4

5

6

7

8

9

10

11

# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

 

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

 

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])



 