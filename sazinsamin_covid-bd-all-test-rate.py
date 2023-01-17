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
import matplotlib.pyplot as plt 
%matplotlib inline
df=pd.read_csv('../input/covid19-bangladesh-dataset/COVID-19-Bangladesh.csv',parse_dates=True)
df.tail(5)
df2=df[df['new_confirmed']>0]
df2.head(5)
test_rate=df2['daily_collected_sample']/df2['new_confirmed']
test_rate
test_rate=test_rate.to_frame()
test_rate.head(7)
test_rate.rename(columns={'0':'test_rate'},inplace=True)
test_rate
test_rate.describe()
mean=test_rate.mean()
plt.figure(figsize=(17,5))
plt.plot(df['date'],df['daily_collected_sample'])
plt.plot(df['date'],df['new_confirmed'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(18,5))
plt.plot(df2['date'],test_rate)
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
df.tail(5)
plt.figure(figsize=(17,5))
plt.plot(df['date'],df['new_confirmed'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
plt.figure(figsize=(17,5))
plt.plot(df['date'],df['total_quarantine'])
plt.plot(df['date'],df['released_from_quarantine'])
plt.xticks(rotation=90)
plt.locator_params(numticks=12)
from sklearn.svm import SVR
le=SVR(kernel='linear')
x_train=df['daily_collected_sample'].values.reshape(-1,1)
y_train=df['new_confirmed'].values.reshape(-1,1)
print(x_train.shape)
print(y_train.shape)
le.fit(x_train,y_train)
print(le.intercept_)
print(le.coef_)
dfx = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_pred.flatten()})
dfx
plt.plot(dfx['Actual'])
plt.plot(dfx['Predicted'])
le.predict([[160000000]])

