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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
df=pd.read_csv("../input/insurance-prediction/insurance.csv")

df.head()
df.shape
df.columns
df.info()
df.describe()
f,axes=plt.subplots(1,2)
sns.kdeplot(df['age'],ax=axes[0])
sns.boxplot(df['age'],ax=axes[1])

f,axes=plt.subplots(1,2)
sns.kdeplot(df['bmi'],ax=axes[0])
sns.boxplot(df['bmi'],ax=axes[1])

f,axes=plt.subplots(1,2)
df.region.value_counts().plot(kind="pie",ax=axes[0])
sns.boxplot(df['region'],ax=axes[1])

f,axes=plt.subplots(1,2)
sns.kdeplot(df['charges'],ax=axes[0])
sns.boxplot(df['charges'],ax=axes[1])

sns.countplot(x="sex",data=df,palette='hls')

sns.countplot(x="smoker",data=df,palette="hls")
sns.countplot(x="children",data=df,palette="hls")
plt.figure(figsize = (14,8))
sns.countplot(x="age",data=df,palette="hls")
corr = df.corr()
ax = sns.heatmap(corr)
# Importing necessary package for creating model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
X = df.drop(['charges'], axis = 1)
y = df.charges

#Normalization
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

#Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=123)
lr = LinearRegression().fit(X_train, y_train)
yhat = lr.predict(X_test)

#Evaluation
r2Score=r2_score(y_test,yhat)
print("R2 Score:",r2Score)
mean_absolute_error=mean_absolute_error(y_test,yhat)
print("Mean Absolute Error:",mean_absolute_error)
mean_squared_error=mean_squared_error(y_test,yhat)
print("Mean Squared Error:",mean_squared_error)