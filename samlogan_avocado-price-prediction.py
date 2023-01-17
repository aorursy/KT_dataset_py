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
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt
data = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

data.head()
data.shape
data = data.rename(columns={'4046':'PLU_4046','4225':'PLU_4225','4770':'PLU_4770'})
data.describe()
object_col = [col for col in data.columns if data[col].dtype=="object" or col=="Unnamed: 0"]

object_col
data_copy = data.copy()

data_copy = data_copy.drop(object_col,axis=1)
data_copy.corr()
X = data_copy.iloc[:,1:-1]

X.head()
y = data_copy.iloc[:,0]

y.head()
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def get_score(n_estimators):

  my_model = RandomForestRegressor(n_estimators,random_state=1)

  my_model.fit(X_train,y_train)

  val_predicted = my_model.predict(X_valid)

  mae_score = mean_absolute_error(y_valid,val_predicted)

  print("MAE: %.4f"%mae_score)

  mse_score = mean_squared_error(y_valid,val_predicted)

  print("MSE: %.4f"%mse_score)

  r_sq_score = r2_score(y_valid,val_predicted)

  print("R2: %.4f"%r_sq_score)
get_score(100)
import xgboost as xgb
xgb = xgb.XGBRegressor()

xgb.fit(X_train,y_train)

val_predicted = xgb.predict(X_valid)
mae_score = mean_absolute_error(y_valid,val_predicted)

print("MAE: %.4f"%mae_score)

mse_score = mean_squared_error(y_valid,val_predicted)

print("MSE: %.4f"%mse_score)

r_sq_score = r2_score(y_valid,val_predicted)

print("R2: %.4f"%r_sq_score)