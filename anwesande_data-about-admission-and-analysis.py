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
df=pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")

df.head()
df.describe()
df.columns
df.corr()
import seaborn as sns
import matplotlib.pyplot as plt

heatmap=sns.heatmap(df.corr(),vmax=1, vmin=-1,annot=True,cmap="BrBG")
plt.figure(figsize=(16, 6))
sns.relplot(x="CGPA",y="GRE Score",data=df)
sns.relplot(x="CGPA",y="TOEFL Score",data=df)
sns.relplot(x="TOEFL Score",y="GRE Score",data=df)
df=df.drop(["Serial No."],axis=1)
df.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X=df.iloc[:,3:-1]

y=df.iloc[:,-1]

X.shape

y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from statsmodels.api import OLS
OLS(y_train,X_train).fit().summary()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df.corr()
df.columns
fig=sns.lmplot(x='CGPA',y='Chance of Admit ',data=df)

fig=sns.lmplot(x='GRE Score',y='Chance of Admit ',data=df)

