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

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats 

from  sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
data = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
data
data.info()
check_norm=data[['Present_Price','Kms_Driven']]
for x in check_norm:

    plt.figure(figsize=(8,3))

    sns.distplot(check_norm[x])
for x in check_norm:

    logs=np.log(check_norm[x])

    plt.figure(figsize=(13,4))

    plt.subplot(1,2,1)

    plt.title(x)

    sns.distplot(logs)

    plt.subplot(1,2,2)

    stats.probplot(logs,dist='norm',plot=plt)

    plt.show()

    
fig,ax=plt.subplots(figsize=(15,10))

sns.boxplot(data=check_norm,ax=ax, fliersize=3)
data_cleaned=data[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
q=data_cleaned['Kms_Driven'].quantile(0.97)

data_cleaned=data_cleaned[data_cleaned['Kms_Driven']<q]

q=data_cleaned['Present_Price'].quantile(0.99)

data_cleaned=data_cleaned[data_cleaned['Present_Price']<q]
data_cleaned.info()
for x in data_cleaned[['Present_Price','Kms_Driven']]:

    plt.figure(figsize=(9,3))

    sns.distplot(data_cleaned[x])
for x in data_cleaned[['Present_Price','Kms_Driven']]:

    logs=np.log(data_cleaned[x])

    plt.figure(figsize=(9,4))

    plt.subplot(1,2,1)

    sns.distplot(logs)

    plt.subplot(1,2,2)

    stats.probplot(logs,dist='norm',plot=plt)

    plt.show()
fig,ax=plt.subplots(figsize=(15,10))

sns.boxplot(data=data_cleaned,ax=ax, fliersize=3)
for x in data_cleaned[['Present_Price','Kms_Driven']]:

    logss_cleaned=np.log(data_cleaned[['Present_Price','Kms_Driven']][x])

    plt.figure(figsize=(15,3))

    plt.subplot(1,2,1)

    sns.distplot(logss_cleaned)

    plt.subplot(1,2,2)

    stats.probplot(logss_cleaned,dist='norm',plot=plt)

    plt.show()

    
data_cleaned['Present_Price']=np.log(data_cleaned['Present_Price'])

data_cleaned['Kms_Driven']=np.log(data_cleaned['Kms_Driven'])
data_cleaned
data_cleaned=pd.get_dummies(data_cleaned,drop_first=True)
data_cleaned=data_cleaned.drop(columns="Fuel_Type_Diesel")
data_cleaned
x=data_cleaned.drop(columns='Selling_Price')

y=data_cleaned['Selling_Price']
scaler=StandardScaler()

X_scaled=scaler.fit_transform(x)
X_scaled
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.25,random_state=42)
model=LinearRegression()

model.fit(X_train,y_train)
ytrain_predict=model.predict(X_train)

ytrain_predict
print("The R square is equal to {}".format(r2_score(y_train,ytrain_predict)))
ytest_predict=model.predict(X_test)

ytest_predict
print("The R square is equal to {}".format(r2_score(y_test,ytest_predict)))
print("The r square value on test data is {}".format(model.score(X_test,y_test)))

print("The r square value on train data is  {}".format(model.score(X_train,y_train)))