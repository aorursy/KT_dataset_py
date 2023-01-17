# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test = pd.read_csv('/kaggle/input/titanic/test.csv')

model = RandomForestRegressor(random_state = 1)

print(data_train.head())

y = data_train.Survived

y = y.fillna(y.mean())

features = ['Pclass', 'Age', 'Fare']

X = data_train[features]

X = X.fillna(X.mean())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model.fit(train_X, train_y)

test_X = data_test[features]

test_X = test_X.fillna(test_X.mean())

print(model.predict(test_X))
#テストデータからpassengerIDをひっぱってくる必要があるので、このcolumnsをいただく

data_test.head()
ID = data_test["PassengerId"]
#また、予測結果を確認し、必要な形に直す。

result = model.predict(test_X)
#しきい値をとりあえず0.5にして、必要な形に直す。

threshold = 0.5

for i in range(len(result)):

    if result[i] < threshold:

        result[i] = 0

    else:

        result[i] = 1



#わしはseries型の方が好きなのでseries型にする。

result = pd.Series(result,dtype="int64")



result.head()
#ここまで出来たら、提出用のデータフレームを作り、csvにアウトプットする。

#一回dict型にする(もっといい方法があるかも)

data = {"PassengerId":ID,"Survived":result}

sub_df = pd.DataFrame(data)
sub_df.head()
#これでcsv形式にする。

sub_df.to_csv("submission.csv",index=False)

#このセルを実行すると、右側のData欄、outputフォルダにcsvファイルが生成される。

#commitをして、完了したらタブを閉じずに、そのままopen versionをクリック

#とんだページの左側、Outputより、Submit to Competitionをクリック(バージジョン確認を忘れずに!)

#しばらくすると点数が出る。