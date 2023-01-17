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
## Gradient Descent Algorithm
import pandas as pd
import numpy as np
import seaborn as sns
np.random.randint(len(df))
df = pd.DataFrame({
    'x': np.random.randint(1, 100, 1000), # B/w 1 to 100, we will 1000 samples
    
})
m = 0.2
c = 10
df['y'] = m * df['x'] + c
#noise = np.random.randint(1,2, len(df))
#df['y'] = df['y'] + noise
sns.scatterplot(data=df, x='x', y='y')
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# x(i) = (x - mean(x))/std(x)
scaler = StandardScaler().fit_transform(df[['x']])
X = pd.DataFrame(scaler, columns=['x'])
X['x'].mean(), X['x'].var()
y = df['y']
#model = LinearRegression().fit(df[['x']], y)
model = LinearRegression().fit(X, y)
model.coef_, model.intercept_
df['yhat'] = model.predict(X)
df['error'] = df['y'] - df['yhat']
sse = np.square(df['error']).sum()
mse = sse / len(df)
print(mse)
X.head()
#m_values = np.random.rand(100)
m_values = np.linspace(-100,100,1000)
c_values = np.random.randint(1, 10, 100)
# yhat = mx + c
errors = []
c = c_values[0]
for m in m_values:
    yhat = m * X['x'] + c
    error = y - yhat
    sse = np.square(error).sum()
    mse = sse / len(X)
    errors.append(mse)
import matplotlib.pyplot as plt
plt.scatter(m_values, errors)
#m_values = np.random.rand(100)
m_values = np.linspace(-100,100,100)
c_values = np.linspace(-50, 50, 100)
# yhat = mx + c
errors = []
#c = c_values[0]
m_all = []
c_all = []
for m in m_values:
    for c in c_values:
        yhat = m * X['x'] + c
        error = y - yhat
        sse = np.square(error).sum()
        mse = sse / len(X)
        errors.append(mse)
        m_all.append(m)
        c_all.append(c)
print(len(errors), len(m_all), len(c_all))

from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection='3d')
ax.scatter3D(m_all, c_all, errors, color='orange')
m = 0
c = 0
lr = 0.001

def calc_mse(y, yhat):
    df = pd.DataFrame({'y': y, 'yhat': yhat})
    df['error'] = df['y'] - df['yhat']
    sse = np.square(df['error']).sum()
    mse = sse / len(y)
    return mse

N = len(y)
errors = []
m_all = []
c_all = []
#mgrad = -2/N sum(xy-mx2-cx)
for i in range(3000):
    yhat = m * X['x'] + c # forward propagation
    error = calc_mse(y, yhat)
    mgrad = (-2/N) * (X['x']*y - m*np.square(X['x']) - c*X['x']).sum()
    cgrad = (-2/N) * (y - m*X['x'] - c).sum()
    
    errors.append(error)
    m_all.append(m)
    c_all.append(c)
    m = m - mgrad * lr
    c = c - cgrad * lr
m, c
plt.plot(errors)
from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection='3d')
ax.scatter3D(m_all, c_all, errors, color='orange')
## Logistic Regression
