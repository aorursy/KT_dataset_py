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
# For this practical example we will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
data = pd.read_csv('/kaggle/input/insurance/insurance.csv')
data.head()
data.describe(include = 'all')
data.isnull().sum()
sns.distplot(data['charges'])
q = data['charges'].quantile(0.99)
data_1 = data[data['charges']<q]
data_1.describe(include = 'all')
sns.distplot(data_1['charges'])
sns.distplot(data_1['bmi'])
q1 = data_1['bmi'].quantile(0.99)
data_2 = data_1[data_1['bmi']<q1]
data_2.describe(include='all')
sns.distplot(data_2['bmi'])
sns.distplot(data_2['children'])
q2 = data_2['children'].quantile(0.99)
data_3 = data_2[data_2['children']<q2]
data_3.describe(include = 'all')
sns.distplot(data_3['children'])
data_3 = data_3.reset_index(drop=True)
data_3.head()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_3['age'],data_3['charges'])
ax1.set_title('age and charges')
ax2.scatter(data_3['bmi'],data_3['charges'])
ax2.set_title('bmi and charges')
ax3.scatter(data_3['children'],data_3['charges'])
ax3.set_title('children and charges')


plt.show()
data_3.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_3[['age','bmi','children']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif
data_with_dummies = pd.get_dummies(data_3, drop_first=True)
data_with_dummies
data_with_dummies.columns.values
data_3.columns.values
cols = ['charges','age', 'bmi', 'children','sex_male', 'smoker_yes',
       'region_northwest', 'region_southeast', 'region_southwest']
data_with_dummies[cols]
y = data_with_dummies['charges']
x = data_with_dummies[['age','bmi', 'children','sex_male', 'smoker_yes']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state = 42)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
reg = LinearRegression()
reg.fit(x_train,y_train)
yhat = reg.predict(x_train)
plt.scatter(yhat,y_train)
from sklearn.metrics import r2_score
r2_score(y_train,yhat)
yhat_test = reg.predict(x_test)
r2_score(y_test,yhat_test)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0,max_depth=5)
regressor.fit(x_train,y_train)
y_pred_dt = regressor.predict(x_test)
r2_score(y_test,y_pred_dt)
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=1000,random_state=0)
regressor_rf.fit(x_train,y_train)
y_pred_rf = regressor.predict(x_test)
r2_score(y_test,y_pred_rf)
