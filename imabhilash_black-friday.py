# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
raw_data=pd.read_csv('../input/BlackFriday.csv')

raw_data.head()
raw_data.describe(include='all')
data=raw_data.copy()
data.head()
data_1=data.replace(np.nan,0)
data_1.head()
data_1.describe(include='all')
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
data_1=data_1.drop(['User_ID'],axis=1)
data_1=data_1.drop(['Product_ID'],axis=1)
data_2=data_1.copy()
data_2['Gender']=data_2['Gender'].map({'F':0,'M':1})

data_2['Age']=data_2['Age'].map({'0-17':0,'18-25':1,'26-35':2,'36-45':3,'46-50':4,'51-55':5,'55+':6})

data_2['City_Category']=data_2['City_Category'].map({'A':0,'B':1,'C':2})

data_2['Stay_In_Current_City_Years']=data_2['Stay_In_Current_City_Years'].map({'0':0,'1':1,'2':2,'3':3,'4':4,'4+':5})
data_2.head()
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
y=data_2['Purchase']

x1=data_2.drop(['Purchase'],axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])]

vif["features"] = x1.columns
vif
x=sm.add_constant(x1)

result_sm=sm.OLS(y,x).fit()

result_sm.summary()
x_train,x_test,y_train,y_test=train_test_split(x1,y,train_size=0.8,random_state=365)
reg=LinearRegression()

reg.fit(x_train,y_train)
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary=pd.DataFrame(x1.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary
y_hat=reg.predict(x_train)
y_hat
sns.distplot(y_hat-y_train)
y_hat_test=reg.predict(x_test)
predicted_purchase=y_hat_test
actual_purchase=y_test.reset_index(drop=True)
results_summary=pd.DataFrame(predicted_purchase,columns=['Predicted Purchase'])

results_summary['Actual Purchase']=actual_purchase
results_summary['Residual']=actual_purchase-predicted_purchase

results_summary['Error']=np.absolute(results_summary['Residual']/actual_purchase)*100
results_summary.sort_values(by='Error')
results_summary.describe(include='all')
sns.distplot(results_summary['Error'])