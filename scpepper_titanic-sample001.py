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
import numpy as np

import pandas as pd
# Data Import

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
# Check Training Data

train.head(10)
# Check Test Data

test.head(10)
# Check Submission Format

gender_submission.head(10)
# Check Data Shape

test_shape = test.shape

train_shape = train.shape

print(test_shape)

print(train_shape)
# Check Data Statistics

train.describe()
# Check Data Statistics

test.describe()
# Check Missing Values

def missing_table(df): 

        null_val = df.isnull().sum()

        percent = 100 * df.isnull().sum()/len(df)

        missing_table = pd.concat([null_val, percent], axis=1)

        missing_table_ren_columns = missing_table.rename(

        columns = {0 : 'missing count', 1 : '%'})

        return missing_table_ren_columns

missing_table(train)
missing_table(test)
# Fill in missing values

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Embarked"] = train["Embarked"].fillna("S")



missing_table(train)
# Convert Data to Number

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S" ] = 0

train["Embarked"][train["Embarked"] == "C" ] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

 

train.head(10)
# Convert Data to Number

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



# Fill in missing values

test["Age"] = test["Age"].fillna(test["Age"].median())

test["Fare"] = test["Fare"].fillna(test["Fare"].median())

 

test.head(10)
# Decision Tree by scikit-learn

from sklearn import tree
target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

 

# 決定木の作成

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

 

# 「test」の説明変数を使って「my_tree_one」のモデルで予測

my_prediction = my_tree_one.predict(test_features)
# 予測データのサイズを確認

my_prediction.shape
#予測データの中身を確認

print(my_prediction)
# PassengerIdを取得

PassengerId = np.array(test["PassengerId"]).astype(int)

 

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

 

# my_tree_one.csvとして書き出し

my_solution.to_csv("tree_submission.csv", index_label = ["PassengerId"])
import keras

from keras.utils.np_utils import to_categorical

# 説明変数と目的変数に分割

keras_y_train = train["Survived"].values



COLUMNS = ["Pclass", "Sex", "Age", "Fare"]

keras_x_train = train[COLUMNS].values

 

keras_x_test = test[COLUMNS].values
import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout

 

# モデル作成

model = Sequential()

model.add(Dense(32, input_shape=(len(COLUMNS),), activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

 

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

 

model.fit(

    keras_x_train, 

    keras_y_train, 

    epochs=30, 

    batch_size=1, 

    verbose=1)

predictions = model.predict(keras_x_test)

 

# テスト値を再読み込みして，クラス分類したカラムを追加

df_out = pd.read_csv("../input/titanic/test.csv")

df_out["Survived"] = np.round(predictions).astype(np.int)

 

# outputディレクトリに出力する

df_out[["PassengerId","Survived"]].to_csv("keras_submission.csv",index=False)