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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df.head()
df.info()
df.describe()
#EDA
#So, there are no missing values to treat...
from matplotlib import pyplot as plt
import seaborn as sns
cols=list(df.columns)
#Check for outliers
plt.figure(figsize=(20,12))
plt.subplot(3,3,1)
sns.boxplot(x=cols[0], data=df)
plt.subplot(3,3,2)
sns.boxplot(x=cols[1], data=df)
plt.subplot(3,3,3)
sns.boxplot(x=cols[2], data=df)
plt.subplot(3,3,4)
sns.boxplot(x=cols[3], data=df)
plt.subplot(3,3,5)
sns.boxplot(x=cols[4], data=df)
plt.subplot(3,3,6)
sns.boxplot(x=cols[5], data=df)
plt.subplot(3,3,7)
sns.boxplot(x=cols[6], data=df)
plt.subplot(3,3,8)
sns.boxplot(x=cols[7], data=df)
plt.subplot(3,3,9)
sns.boxplot(x=cols[8], data=df)
plt.show()
df.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
#Drop unnecessary columns
df.drop('Serial No.',axis=1 ,inplace=True)
print(df)
# Correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True)
plt.show()
sns.pairplot(data=df,hue='Research',markers=["^", "v"],palette='inferno')
plt.show()
y=df.pop(cols[-1])
X=df
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit_transform(X_train,y_train)
# Build model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
preds=lr.predict(X_test)
from sklearn.metrics import accuracy_score,mean_squared_error

rms=np.sqrt(mean_squared_error(y_test, preds))
rms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
regressors=[['Linear Regression :',LinearRegression()],
       ['Decision Tree Regression :',DecisionTreeRegressor()],
       ['Random Forest Regression :',RandomForestRegressor()],
       ['Gradient Boosting Regression :', GradientBoostingRegressor()],
       ['Ada Boosting Regression :',AdaBoostRegressor()],
       ['Extra Tree Regression :', ExtraTreesRegressor()],
       ['K-Neighbors Regression :',KNeighborsRegressor()],
       ['Support Vector Regression :',SVR()]]
reg_pred=[]
print('Results...\n')
for name,model in regressors:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    rms=np.sqrt(mean_squared_error(y_test, predictions))
    reg_pred.append(rms)
    print(name,rms)