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
#supress warning

import warnings
warnings.filterwarnings("ignore")
#save file path
real_estate_file_path = '/kaggle/input/real-estate-price-prediction/Real estate.csv'
#read data and store data
real_estate = pd.read_csv(real_estate_file_path)
#summary of real_estate data
real_estate.head()
#inspect various aspect of dataframe

real_estate.shape
real_estate.info()
#to check the null values
real_estate.isnull().sum()
#describe the data
real_estate.describe()
#There is no need of 'No' column and 'Date' column, hence dropping it.
real_estate.drop(['No'], axis=1, inplace=True)
real_estate.drop(['X1 transaction date'],axis=1, inplace=True)
#check dataset after dropping 'No' col
real_estate.head()
#import libraries
import matplotlib.pyplot as plt
import seaborn as sns
#Visualising all numeric variable
plt.figure(figsize=(6,12))
sns.pairplot(real_estate)
plt.show()
#Find correlation
plt.figure(figsize=(6,6))
sns.heatmap(real_estate.corr(),annot=True)
#import libraries
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train,df_test = train_test_split(real_estate, train_size=0.70, test_size=0.30,random_state=100)
print(df_train.head())
print(df_test.head())
#Dividing X and y sets for model building
y_train = df_train.pop('Y house price of unit area')
X_train = df_train
print(y_train.head())
print(X_train.head())
#import Linear regression
from sklearn.linear_model import LinearRegression
#fit the model
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_
#The coefficient of all independent variable are as follows
coeff = pd.DataFrame(lm.coef_, X_train.columns, columns=['coefficient'])
coeff
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train)
lm_1 = sm.OLS(y_train, X_train).fit()
print(lm_1.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_price = lm_1.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_true=y_train,y_pred=y_train_price)
#plot histogram of error terms
fig = plt.figure()
sns.distplot((y_train-y_train_price), bins=20)
fig.suptitle('Error Terms',fontsize = 20)
plt.xlabel('Error',fontsize=17)
y_test = df_test.pop('Y house price of unit area')
X_test = df_test
y_test_pred = lm_1.predict(X_test)
#import library
from sklearn.metrics import r2_score
#Evaluate r2
r2_score(y_true=y_test,y_pred=y_test_pred)
df = pd.DataFrame({'Actual':y_test,'Predictions':y_test_pred})
df['Predictions']= round(df['Predictions'])
df.head()
sns.regplot('Actual','Predictions',data=df)
from sklearn import metrics

#Mean absolute error(MAE)
print('MAE',metrics.mean_absolute_error(y_test,y_test_pred))
#Mean squared error(MSE)
print('MSE',metrics.mean_squared_error(y_test,y_test_pred))
#Root mean squared error(RMSE)
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,y_test_pred)))