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
import numpy as np
import pandas as pd
path = '../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
df = pd.read_csv(path)
df.head()
df.describe()
df['salary'].isna().sum()
df['salary']=df['salary'].replace(np.nan,df['salary'].mean())
df.describe()
df.head()
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
lm=LinearRegression()
z=df[['ssc_p','hsc_p']]
y=df['mba_p']
lm.fit(z,y)
y_pred=lm.predict(z)
print("The coefficients are: ",lm.coef_," and the intercept is: ",lm.intercept_)
y_pred[0:4]
plt.figure(figsize=(12,10))
ax1= sns.distplot(y,color='blue',hist=False,label="Actual value")
sns.distplot(y_pred,hist=False,color='red',label="Predicted value",ax=ax1)
plt.show()
df[['ssc_p','hsc_p','mba_p']].corr()
lm.score(z,y)
#from sklearn.metrics import mean_squared_error
#mean_squared_error(y,y_pred)
#MOdel 1

z1= df[['ssc_p','degree_p']]
y=df['mba_p']
lm1=LinearRegression()
lm1.fit(z1,y)
y1_pred=lm1.predict(z1)
print("The coefficients are: ",lm1.coef_," and the intercept is: ",lm1.intercept_)
score1= lm1.score(z1,y)
print(score1)
plt.figure(figsize=(12,10))
ax1=sns.distplot(y,hist=False,color='blue',label="Actual Value")
sns.distplot(y1_pred,hist=False,color='red',label="Predicted values",ax=ax1)
plt.show()
#Model 2

z2=df[['hsc_p','degree_p']]
y=df['mba_p']
lm2=LinearRegression()
lm2.fit(z2,y)
y2_pred=lm2.predict(z2)
score2=lm2.score(z2,y)
print(score2)
ax1=sns.distplot(y,hist=False,color='blue',label='Actual Values')
sns.distplot(y2_pred,hist=False,color='red',label='Predicted values',ax=ax1)
#Model 3
z3=df[['ssc_p','hsc_p','degree_p']]
y=df['mba_p']
lm3=LinearRegression()
lm3.fit(z3,y)
y3_pred=lm3.predict(z3)
lm3.score(z3,y)
ax1=sns.distplot(y,hist=False,color='blue',label='Actual values')
sns.distplot(y3_pred,hist=False,color='red',label='Predicted values',ax=ax1)
#Splitting the data into train and test data with 80:20 ratio and fitting train data to make the model then predicting 
#using test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(z3,y,test_size=0.2,random_state=0)
lre=LinearRegression()
lre.fit(x_train,y_train)
pred=lre.predict(x_test)
ax1=sns.distplot(y_test,hist=False,color='blue',label='Actual Value')
sns.distplot(pred,hist=False,color='red',label='Predicted values',ax=ax1)

data=pd.DataFrame({'Actual mba_p':y_test,'Predicted mba_p':pred})
data.head()
plt.figure(figsize=(7,7))
sns.countplot(df['gender'],color='red')
plt.show()
