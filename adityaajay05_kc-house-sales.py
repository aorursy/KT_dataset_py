# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.describe()
df.head(10)
df.head(10)
df.mode()
np.shape(df)
hm = pd.DataFrame(df.drop(['date', 'id'], axis = 1).values)
corrmat =  hm.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(hm[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.head()
X = pd.DataFrame(df.drop(['date','id', 'price'], axis = 1).values)
y = df.iloc[:, 2].values
df['date'] = df['date'].str[:8]
print(df['date'])
X['Days Since'] = (pd.to_datetime('20200703') - pd.to_datetime(df['date'])).dt.days
print(X['Days Since'])
X.head()
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 42)
dtr.fit(X_train, y_train)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 42)
rfr.fit(X_train, y_train)
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)
y_pred_rfr = rfr.predict(X_test)
y_pred_xgb = xgb.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_dtr)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_rfr)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred_xgb)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
