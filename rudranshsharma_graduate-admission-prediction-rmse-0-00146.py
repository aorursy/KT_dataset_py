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
data1 = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col = "Serial No.")
X = data1.drop(["Chance of Admit "], axis = 1)
y = data1["Chance of Admit "]
data1
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.heatmap(data1.corr(), annot = True, )
data = pd.melt(data1,id_vars="Chance of Admit ",
               var_name="features",
                    value_name='value')
_ = plt.subplots(figsize= (10, 5))
sns.boxplot(x = "features", y = "value",  data = data)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)
my_mod = XGBRegressor(max_depth = 1, n_estimators = 100, min_child_weight = 0).fit(X_train, y_train)
print(rmse(y[400:], my_mod.predict(X[400:])))
print(mae(y_valid, my_mod.predict(X_valid)))


from sklearn.model_selection import GridSearchCV
params = {"min_child_weight" : list(range(10))}
search = GridSearchCV(my_mod, param_grid = params, cv = 3, n_jobs = -1).fit(X_train, y_train)
search.best_params_
