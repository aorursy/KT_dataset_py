# データを読み込み、分類、モデル構築とEDAを行うためのライブラリーインポート

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)



# 極端な外れ値の座標または負の運賃のデータを削除する

data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +

                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +

                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +

                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +

                  'fare_amount > 0'

                  )



y = data.fare_amount



base_features = ['pickup_longitude',

                 'pickup_latitude',

                 'dropoff_longitude',

                 'dropoff_latitude',

                 'passenger_count']



X = data[base_features]





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)



# フィードバックシステムのための環境セットアップ

from learntools.core import binder

binder.bind(globals())

from learntools.ml_explainability.ex2 import *

print("Setup Complete")



# データを見る

print("Data sample:")

data.head()
train_X.describe()
train_y.describe()
# あなたの答えを確認しよう (このコードを走らせてみてください！)

q_1.solution()
import eli5

from eli5.sklearn import PermutationImportance



# この問題で使用するには、以下のコードを少し変更してください。 

#perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)



# あなたの答えを確認しよう

q_2.check()



# 次の行のコメントアウトを解除して、結果を視覚化します

#eli5.show_weights(perm, feature_names = val_X.columns.tolist())
# q_2.hint()

# q_2.solution()
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
# create new features

data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)

data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)



features_2  = ['pickup_longitude',

               'pickup_latitude',

               'dropoff_longitude',

               'dropoff_latitude',

               'abs_lat_change',

               'abs_lon_change']



X = data[features_2]

new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)

second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)



# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y

# Use a random_state of 1 for reproducible results that match the expected solution.

perm2 = ____



# show the weights for the permutation importance you just calculated

____



# Check your answer

q_4.check()
# Check your answer (Run this code cell to receive credit!)

q_4.solution()
# Check your answer (Run this code cell to receive credit!)

q_5.solution()
# Check your answer (Run this code cell to receive credit!)

q_6.solution()