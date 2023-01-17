# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
#np.random.seed(1)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
###-----------------------------------###
df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv') 
game_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv') 
print("df.shape:", df.shape)
print("game_df.shape:", game_df.shape)
###-----------------------------------###
df[df.isnull()].sum()
df['winPlacePerc'][df['winPlacePerc'].isnull()]=0
x_name = df[df.columns[:3]]
x_data = df[df.columns[3:-1]]
y_data = df[df.columns[-1]]
del x_data['matchType']
x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = y_data.reshape(-1,1)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
x_train.shape
y_train.shape
x_train_30 = x_train[:100000]
y_train_30 = y_train[:100000]
print("x_train_30:", x_train_30.shape)
print("y_train_30:", y_train_30.shape)
###-----------------------------------###
rf_model = RandomForestRegressor()
rf_model.fit(x_train_30, y_train_30)
y_predict = rf_model.predict(x_val)
print("\n預測比賽成績：\n",y_predict)

# MeanSquaredError
print("\n均方誤差為：", mean_absolute_error(y_val, y_predict))
game_df[game_df.isnull()].sum()
game_df_name = game_df[game_df.columns[:3]]
game_df_name.head()
game_x_data = game_df[game_df.columns[3:]]
del game_x_data['matchType']
game_y_predict = rf_model.predict(game_x_data)
submit = pd.DataFrame()
submit['Id'] = game_df_name['Id']
submit['winPlacePerc'] = game_y_predict
submit.to_csv('submission.csv', index=False)