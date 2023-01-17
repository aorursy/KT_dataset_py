# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv', delimiter=',', low_memory=True)
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv', delimiter=',', low_memory=True)
train.ID_code = train.ID_code.astype('category')
test.ID_code = test.ID_code.astype('category')
train.ID_code = train.ID_code.cat.codes
test.ID_code = test.ID_code.cat.codes
x_train, y_train = train.drop(['target'], axis=1), train.target

# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)
# スコアを表示
print(m.score(x_train, y_train))
# 作成したランダムフォレストのモデル「m」に「test」を入れて予測する
preds = m.predict(test)

# 予測値 predsをnp.exp()で処理
np.exp(preds)

# Numpy配列からpandasシリーズへ変換
preds = pd.Series(np.exp(preds))

# テストデータのIDと予測値を連結
submit = pd.concat([test.ID_code, preds], axis=1)

# カラム名をつける
submit.columns = ['ID_code', 'preds']

# 提出ファイルとしてCSVへ書き出し
submit.to_csv('submit.csv', index=False)