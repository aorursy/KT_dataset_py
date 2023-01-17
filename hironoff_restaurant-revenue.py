# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
"""
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#解凍
import zipfile

with zipfile.ZipFile('../input/restaurant-revenue-prediction/test.csv.zip') as existing_zip:
    existing_zip.extractall('data')

with zipfile.ZipFile('../input/restaurant-revenue-prediction/train.csv.zip') as existing_zip:
    existing_zip.extractall('data')
    
train_df = pd.read_csv('data/train.csv')
train_df.info()  #nullはなし

test_df = pd.read_csv('data/test.csv')

#train_df.set_index('Id')

train_df.head()
#とりあえず意味のないカラムを削除し、train_yを分離
train_y = train_df['revenue'] #train_yはlistではなくseries!!

train_x = train_df.drop(['revenue', 'Id'], axis = 1)

#一回のみ実行
train_x = train_x.drop('Open Date', axis = 1)

for label in ['City', 'City Group', 'Type']:
    le = LabelEncoder() 
    le.fit(train_x[label])
    train_x[label] = le.transform(train_x[label])

train_x
train_x.nunique() #ユニークな値の数
#import関連
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
X = train_x[:100]
Y = train_y[:100]
Xt = train_x[100:]
Yt = train_y[100:]
stime = time.time()

rfc = RandomForestClassifier()
rfc.fit(X, Y)

Y_pred = rfc.predict(Xt)
RMSE = mean_absolute_error(Y_pred, Yt)
print(f'RMSE: {RMSE}')

etime = time.time()

print('経過時間：' + str(etime - stime))
Yt.values
#結果がクラスじゃなくてloglossなのでわかりづらい？
Y_pred
