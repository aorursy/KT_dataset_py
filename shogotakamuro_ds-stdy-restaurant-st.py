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
# ライブラリのインポート



import pandas as pd



from sklearn.ensemble import RandomForestRegressor



# データの読み込み



train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')



test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')
#目的変数を抽出



train_y = train["revenue"]



#説明変数を抽出（Id、カテゴリ変数、目的変数を除き、数値変数のみを抽出）



train_X_columns = [col for col in train.columns if col not in ["Id", "Open Date","City","City Group","Type","revenue"]]



#RandomForestで学習



rf = RandomForestRegressor(random_state=0)



rf.fit(train[train_X_columns], train_y)
# テストデータにて予測



prediction = rf.predict(test[train_X_columns])



prediction
# 予測した値を提出用CSVファイル(submissionファイル)に書き出し



submission = pd.DataFrame({"Id":test.Id, "Prediction":prediction})



submission.to_csv("submission.csv", index=False)