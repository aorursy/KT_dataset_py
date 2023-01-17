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
train = pd.read_csv("/kaggle/input/st4035-2020-inclass-1/train_data.csv")
test = pd.read_csv("/kaggle/input/st4035-2020-inclass-1/test_data.csv")
train.shape
duplicate_rows_train = train[train.duplicated()]
print("number of duplicate rows: ", duplicate_rows_train.shape)
train.shape
train['X2'] = np.power(train['X2'], 3)
test['X2'] = np.power(test['X2'], 3)

train['X4'] = np.power(train['X4'], 4)
test['X4'] = np.power(test['X4'], 4)
from scipy import stats
import numpy as np

#Using Z-Scores to remove outliers
z_train = np.abs(stats.zscore(train))
z_test = np.abs(stats.zscore(test))

train = train[(z_train < 3).all(axis=1)]
test = test[(z_test < 3).all(axis=1)]
# Seperating into independent and dependent variables
X_train = train.iloc[:, 1:-1]
Y_train = train.iloc[:,-1]

X_test = test.iloc[:, 1:]
# Standardizing the features of train and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_sd = sc.fit_transform(X_train)
X_test_sd = sc.transform(X_test)

X_train = pd.DataFrame(X_train_sd, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(X_test_sd, index=X_test.index, columns=X_test.columns)
#from sklearn.ensemble import RandomForestRegressor
#model = RandomForestRegressor(n_estimators=30, random_state=0)
#model.fit(X_train, Y_train)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)

#from sklearn.neural_network import MLPRegressor
#model = MLPRegressor(random_state=0, max_iter=5000)
#model.fit(X_train, Y_train)

#from sklearn.ensemble import GradientBoostingRegressor
#model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls').fit(X_train, Y_train)
#model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
# Including the ID column to the output dataframe
Y_pred_df = pd.DataFrame({"ID":test['ID'], "X5":Y_pred })
Y_pred_df
Y_pred_df.to_csv('submission.csv', index=False)
X_train
