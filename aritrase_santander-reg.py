# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train rows and columns : ", train_df.shape)
print("Test rows and columns : ", test_df.shape)
y = train_df['target'].copy()
X = train_df.drop(labels=['target','ID'],axis=1)
X_test = test_df.drop(labels=['ID'],axis=1)
#X.head()
print(type(y))
print(X.shape)
print(X_test.shape)
#
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns
mis_val_table_ren_columns = missing_values_table(X)
#print(mis_val_table_ren_columns)
nan_col = list(mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values']> 95].index)
print(nan_col) # no missing values column
# 256 cols with no variation
for col in X.columns.values:
    if(len(X[col].unique()) == 1):
        nan_col.append(col)
print(len(nan_col))
# Drop these columns
X.drop(nan_col,inplace = True ,axis=1)
X_test.drop(nan_col,inplace = True ,axis=1)
print(X.shape)
print(X_test.shape)
print(type(X))
print(type(X_test))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

color = sns.color_palette()
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
plt.figure(figsize=(12,8))
sns.distplot( y, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()
print(y)
y = np.log1p(y)
print(y)
plt.figure(figsize=(12,8))
sns.distplot( y, bins=50, kde=False)
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()
# Feature Scaling - StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
"""# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#y_pred = np.expm1(y_pred)"""
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape)
from sklearn.metrics import mean_squared_log_error,mean_squared_error
#from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=400, n_jobs=-1,oob_score = True,random_state =1, max_depth=8,
                                 max_features = "auto",verbose=1,bootstrap=True,max_leaf_nodes=31)
rf_model.fit(X_train, y_train)
RMSLE=np.sqrt(mean_squared_error(y_val,rf_model.predict(X_val)))
print(RMSLE)
pred_rf=np.expm1(rf_model.predict(X_test))
pred_rf
# Making a submission file #
sub_df = pd.DataFrame({"ID":test_df["ID"].values})
sub_df["target"] = pred_rf
sub_df.to_csv("submission_rf.csv", index=False)
