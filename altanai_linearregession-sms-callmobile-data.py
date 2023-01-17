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
%matplotlib inline

df = pd.read_csv("/kaggle/input/mobile-phone-activity/sms-call-internet-mi-2013-11-01.csv")
df.info()
df.head()
df.describe()
df.corr()
sns.jointplot(data=df,x="smsin", y="smsout")
sns.jointplot(data=df,x="callin",y="callout")
sns.heatmap(df.corr())
sns.pairplot(df)
grid = sns.PairGrid(df)
grid.map(plt.scatter)
# Based on above plots we can see a good coorectaion between callin and sms in 
# Linear model plot 

sns.lmplot(data=df, x="smsin", y="callin")
sns.lmplot(data=df, x="callin", y="callout")
df.columns

df.count

# You can see a lot of problem with Nan in train model as shown below 
# # df["CellID"] = df["CellID"].fillna(0)
# df['CellID'].dropna().unique()
# df['smsin'].dropna().unique()
# df['smsout'].dropna().unique()
# print(df)


# making new data frame with dropped NA values 
df1 = df.dropna(axis = 0, how ='any') 
  
# comparing sizes of data frames 
print("Old data frame length:", len(df), "\nNew data frame length:",  
       len(df1), "\nNumber of rows with at least 1 NA value: ", 
       (len(df)-len(df1))) 
df1
# Training a Linear Regression Map 

# X and Y coords

X = df1[['CellID' , 'internet' , 'smsin' , 'countrycode' , 'callin' , 'callout']]
y = df1['smsout']
# train test the split 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# creating and tarning the model 

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
# Model Evluation 

# print the intercept
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
# Prediction from the model 

predictions = lm.predict(X_test)

# predictios in scatter plot 
plt.scatter(y_test,predictions)

# the straighter the line and better the production, lof of noise in these preditcions 
sns.distplot((y_test-predictions),bins=50);
# Regression Evaluartion Metrics 

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
metrics.explained_variance_score(y_test, predictions)
# Residuals 

sns.distplot((y_test-predictions) , bins=10)
# Coeff 

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf
# smsin	0.731425 is highest association in sms out 
