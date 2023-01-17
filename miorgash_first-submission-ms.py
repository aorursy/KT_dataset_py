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
# data wrangling

import pandas as pd



# visualization

from IPython.core.display import display



# modeling

from sklearn.svm import SVC



# データ読み込み

train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")



# モデル入力データを作成

X_train = train_df.loc[:, ['Fare', 'Age']].fillna(0)

y_train = train_df.loc[:, ['Survived']]

X_test = test_df.loc[:, ['Fare', 'Age']].fillna(0)



# とりあえずモデルを作る

model = SVC(random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

submission_df = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

submission_df.to_csv('./submission.csv', index=False)