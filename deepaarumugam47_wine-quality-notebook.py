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
df=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.columns
df.iloc[100:110]
df.info()
df.dtypes
df.quality.unique()
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(14,8)})
sns.countplot(df['quality'])
sns.pairplot(df)
df.hist(bins=10, figsize=(16,12))
plt.show()
# Creating pivot table for red wine
columns = list(df.columns).remove('quality')
df.pivot_table(columns, ['quality'], aggfunc=np.median)    # By default the aggfunc is mean
df.corr()
# red wines
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(), cmap='bwr', annot=True)     # annot = True: to display the correlation value in the graph
x=df.drop(['quality'],axis=1)
y=df.quality
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
np.shape(x_train)
np.shape(x_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred=np.round(y_pred,0)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: ",mean_squared_error(y_test, y_pred, squared=False))
confusion_matrix(y_test,y_pred)