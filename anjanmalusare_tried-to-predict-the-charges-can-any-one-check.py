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
#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")
data
data.describe(include="all")
sns.distplot(data["children"])
q = data["children"].quantile(0.99)
data = data[data["children"]<q]
sns.distplot(data["children"])
data.describe(include="all")
q = data["charges"].quantile(0.99)
data = data[data["charges"]<q]
sns.distplot(data["charges"])
data.reset_index(drop = True, inplace = True)
data.describe()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# the target column (in this case 'weight') should not be included in variables
#Categorical variables already turned into dummy indicator may or maynot be added if any
variables = data[["age", "bmi", "children"]]
X = add_constant(variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]
vif['features'] = X.columns
vif
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(data['age'], data['charges'])
ax1.set_title('age and charges')

ax2.scatter(data['bmi'], data['charges'])
ax2.set_title('bmi and charges')

ax3.scatter(data['children'], data['charges'])
ax3.set_title('children and charges')
data['log_charges'] = np.log(data['charges'])
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(data['age'], data['log_charges'])
ax1.set_title('age and charges')

ax2.scatter(data['bmi'], data['log_charges'])
ax2.set_title('bmi and charges')

ax3.scatter(data['children'], data['log_charges'])
ax3.set_title('children and charges')
data['log_age'] = np.log(data['age'])
data['log_bmi'] = np.log(data['bmi'])
data['log_children'] = np.log(data["children"])
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(data['log_age'], data['log_charges'])
ax1.set_title('age and charges')

ax2.scatter(data['log_bmi'], data['log_charges'])
ax2.set_title('bmi and charges')

ax3.scatter(data['log_children'], data['log_charges'])
ax3.set_title('children and charges')
data = data.drop(["age", "bmi", "children"], axis = 1)
data = pd.get_dummies(data, drop_first = True)
data.describe()

data[data['log_children'].apply(lambda x: x < 1000)].sort_values('log_children', ascending = True)
data=data.replace([np.inf, -np.inf], np.nan)
data.isnull().sum()
data.dropna(inplace=True)


data.reset_index(drop = True, inplace = True)
#Declaring independent variable i.e x
#Declaring Target variable i.e y
y = data['log_charges']
x = data.drop(['log_charges'], axis = 1)
scaler = StandardScaler() #Selecting the standardscaler
scaler.fit(x)#fitting our independent variables
scaled_x = scaler.transform(x)#scaling#Splitting our data into train and test dataframe
x_train,x_test, y_train, y_test = train_test_split(scaled_x, y , test_size = 0.2, random_state = 47)
reg = LinearRegression()#Selecting our model
reg.fit(x_train,y_train)
y_hat= reg.predict(x_train)
fig, ax = plt.subplots()
ax.scatter(y_train, y_hat)
#Residual graph
sns.distplot(y_train - y_hat)
plt.title('Residual Graph')
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
#Predicting with x_test
y_hat_test = reg.predict(x_test)
reg.score(x_test, y_test)
plt.scatter(y_test, y_hat_test, alpha=0.5)
plt.show()

summary = pd.DataFrame( data = x.columns.values, columns = ['Features'] )
summary['charges'] = reg.coef_
summary
#Creating a new dataframe
df1 = pd.DataFrame( data = np.exp(y_hat_test), columns = ['Predictions'] )
#Resetting index to match the index of y_test with that of the dataframe
y_test = y_test.reset_index(drop = True)
#target column will hold our predicted values
df1['target'] = np.exp(y_test)
#Substrating predictions from target to get the difference in value
df1['Residual'] = df1['target'] - df1['Predictions']

#Difference in percentage
df1['Difference%'] = np.absolute(df1['Residual']/ df1['target'] * 100)
df1.describe()

