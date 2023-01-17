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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

XGBmodel_1 = XGBRegressor(n_estimators=1000, learning_rate=0.5)
#XGBmodel_2 = XGBRegressor(n_estimators=250, booster='dart')
jackglobe_data = pd.read_csv("/kaggle/input/JacksonvilleGlobal_train.csv")
features = ['Year','Global Year Avg (F)']
X = jackglobe_data[features]
y = jackglobe_data['Jackson Year Avg (F)']
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
jackglobe_data.describe()
jackglobe_test = pd.read_csv("/kaggle/input/JacksonvilleGlobal_test.csv")
jacktest = jackglobe_test.rename(columns={'year':'Year'})
jacktest = jackglobe_test.rename(columns={'jacksonville avg temp': 'Jackson Year Avg (F)'})
jacktest = jackglobe_test.rename(columns={'global avg temp':'Global Year Avg (F)'})
jacktest.describe()
jacktest = jacktest.rename(columns={'jacksonville avg temp': 'Jackson Year Avg (F)'})
jacktest = jacktest.rename(columns={'year':'Year'})
jacktest.describe()
features = ['Year', 'Global Year Avg (F)']
X_test = jacktest[features]
XGBmodel_1.fit(X_train, y_train,
           early_stopping_rounds=5,
           eval_set=[(X_val, y_val)])
# Make 1st prediction round
val_pred1 = XGBmodel_1.predict(X_val)
#pred2 = XGBmodel_2.predict(X_val)

# Get MAE for prediction
mae1 = mean_absolute_error(val_pred1, y_val)
#mae2 = mean_absolute_error(pred2, y_val)
print(f"MAE1 for pred1: {mae1}")
#print(f"MAE2 for pred2: {mae2}")
test_pred1 = XGBmodel_1.predict(X_test)

output = pd.DataFrame({'Year': X_test.Year,'Jacksonville prediction temp':test_pred1})
output.to_csv('test_pred1.csv')
