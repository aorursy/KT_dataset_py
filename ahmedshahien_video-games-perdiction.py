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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error 


data = pd.read_csv('../input/videogamesales/vgsales.csv')
df=pd.DataFrame(data)

data.head()

data.info()
df.dropna(how='any',inplace=True)
data.info()
data.describe()
data.columns.values
data['Name'].unique()
data['Genre'].unique()
sns.distplot(data['Global_Sales'])
data = pd.concat([data['Global_Sales'], data['Year']], axis=1)
data.plot.scatter(x='Year', y='Global_Sales', ylim=(0,10));
#scatterplot

sns.set()
cols = ['Year','NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sns.pairplot(df[cols], size = 2.5)
plt.show();
le = LabelEncoder()

le.fit(df['Name'])
le.transform(df['Name'])
#print(le.transform(df['Name']))
df.head()
le = LabelEncoder()

le.fit(df['Platform'])
le.transform(df['Platform'])
le = LabelEncoder()

le.fit(df['Genre'])
le.transform(df['Genre'])
 #correlation matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);
sns.heatmap(corrmat,fmt=".2f",cmap='coolwarm', annot=True)
le = LabelEncoder()

 
df["le_Platform"] = le.fit_transform(df["Platform"])
df["le_Name"] = le.fit_transform(df["Name"])
df["le_Genre"] = le.fit_transform(df["Genre"])
 
X=df.drop(['Publisher','Global_Sales','Rank','Platform','Name','Genre'],axis=1)
y=df['Global_Sales']
#Standard Scaler for Data

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

 
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
 
#Applying Ridge Regression Model 

'''
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False,
                           copy_X=True, max_iter=None, tol=0.001, solver='auto',
                           random_state=None)
'''

RidgeRegressionModel = Ridge(alpha=1.0,random_state=33)
RidgeRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X_train, y_train))
print('Ridge Regression Test Score is : ' , RidgeRegressionModel.score(X_test, y_test))
#Calculating Prediction
y_pred = RidgeRegressionModel.predict(X_test)
print('Predicted Value for Ridge Regression is : ' , y_pred[:10])
 
 
 
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Absolute Error Value is : ', MAEValue)