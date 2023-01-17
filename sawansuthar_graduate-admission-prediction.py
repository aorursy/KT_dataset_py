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

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df = df.drop(axis = 1,columns = "Serial No.")
df.columns
print(df.shape)

df.head(10)
df.isnull().sum()
df.describe()
sns.pairplot(data=df,x_vars = df.drop(axis=1,columns = "Chance of Admit ").columns,y_vars = ["Chance of Admit "])
X = df.drop(axis=1,columns = "Chance of Admit ")

Y = df["Chance of Admit "]
list_of_cat_columns = [2,3,4,6]

for i in list_of_cat_columns:

    le = LabelEncoder()

    X.iloc[:,i] = le.fit_transform(X.iloc[:,i])



X_ = pd.get_dummies(X,columns = ['University Rating', 'SOP', 'LOR ',

       'Research'])
plt.figure(figsize=(20,15))

sns.heatmap(X_.corr())
#X__ = np.append(arr = np.ones((400,1)).astype(int),values=X_,axis=1)
import statsmodels.api as sm

ols = sm.OLS(endog=Y,exog=X_).fit()

ols.summary()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_= sc.fit_transform(X_)
X_val = X_[400:500]

X_tr=X_[0:400]

y_val = Y[400:500]

y_tr=Y[0:400]
x_train,x_test,y_train,y_test = train_test_split(X_tr,y_tr,train_size=0.8)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

print(lr.score(x_train,y_train))

print(lr.score(x_test,y_test))
prediction=lr.predict(x_test)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(prediction,y_test))
from sklearn.tree import DecisionTreeRegressor
model_2 = DecisionTreeRegressor(random_state=2,max_depth=7,splitter = "best",criterion = "mse")

model_2.fit(x_train,y_train)

print(model_2.score(x_train,y_train)*100)

print(model_2.score(x_test, y_test)*100)
from sklearn.ensemble import RandomForestRegressor
model_3 = RandomForestRegressor(max_depth = 8,random_state=2)

model_3.fit(x_train,y_train)

print(model_3.score(x_train,y_train)*100)

print(model_3.score(x_test, y_test)*100)
from sklearn.svm import SVR
# model_4 = SVR()

# model_4.fit(np.delete(x_train,[4,12,20,26],axis=1),y_train)

# print(model_4.score(np.delete(x_train, [4,12,20,26],axis=1),y_train)*100)

# #print(model_4.score(x_test, y_test)*100)
# print(model_4.score(np.delete(x_test, [4,12,20,26],axis=1), y_test)*100)
print(mean_squared_error(y_val,lr.predict(X_val)))
print(mean_squared_error(y_val,model_3.predict(X_val)))
df_1 = df[400:500]
df_1 = df_1.reset_index(drop=True)
df_1
predict_val = pd.DataFrame(lr.predict(X_val),columns = ["prediction"])
result = pd.concat([df_1, predict_val], axis=1)
result