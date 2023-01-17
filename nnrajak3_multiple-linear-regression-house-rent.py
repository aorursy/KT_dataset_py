# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
df.info()
df.head(3)
df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
df['floor'] = df['floor'].fillna(df['floor'].median()).astype(int)
import seaborn
plt.figure(figsize=(5, 3))
plt.tight_layout()
seaborn.distplot(df['total (R$)'])
plt.figure(figsize=(5, 3))
df['total (R$)'].plot.box(grid=True)
rows = df[df['total (R$)']>50000].index
df.drop(rows, inplace=True)
plt.figure(figsize=(5, 3))
plt.tight_layout()
seaborn.distplot(df['total (R$)'])
plt.figure(figsize=(5, 3))
df['total (R$)'].plot.box(grid=True)
df.head(3)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
X[:, 6] =le.fit_transform(X[:, 6])
le = LabelEncoder()
X[:, 7] =le.fit_transform(X[:, 7])
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
transformer = ColumnTransformer([('city', OneHotEncoder(drop=[0]), [0])], remainder='passthrough')
X = transformer.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
act_vs_pred = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = act_vs_pred[:20]
df1.plot(kind='bar')
plt.show()
from sklearn.metrics import r2_score, mean_squared_error
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2:', r2_score(y_test, y_pred))
residuals = y_test - y_pred
seaborn.distplot(residuals)
import scipy as sp
fig, ax = plt.subplots()
_, (_, _, r) = sp.stats.probplot(residuals, plot=ax, fit=True)
np.mean(residuals) # mean of the residuals is near to 0
import statsmodels.api as sm
acf = sm.graphics.tsa.plot_acf(residuals)
acf.show()
# No auto correlation of residuals
