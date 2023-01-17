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
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv')
row,column = df.shape
print(f"row: {row},column: {column}")
df.head()
df.isnull().sum()
df['model'].value_counts()
model_count=df['model'].value_counts()
low_mod = model_count[model_count <= 50]
low_mod
df['model'] = df['model'].apply(lambda y : 'Others' if y in low_mod else y)
df['model'].value_counts()
df.head()
df['transmission'].value_counts()
df['fuelType'].value_counts()
df['engineSize'].describe()
df['engineSize'].value_counts()
df = df[(df['engineSize'] > 1.0)]
df.shape
df.mpg.describe()
df.head()
u_bound=df['mpg'].quantile(0.95)
l_bound=df['mpg'].quantile(0.05)
print(u_bound,l_bound)
df = df.loc[df['mpg']<u_bound]
df
df = df.loc[df['mpg']>l_bound]
df.head()
df['tax'].describe()
df['tax'].sort_values()
df.shape
fuel_dummies=pd.get_dummies(df['fuelType'])
fuel_dummies.shape
df = pd.concat([df,fuel_dummies],axis=1)
print(df.shape)
transmission_dummies = pd.get_dummies(df['transmission'])
model_dummies = pd.get_dummies(df['model'])
df = pd.concat([df,transmission_dummies,model_dummies],axis=1)
df
df = df.drop(['Others'],axis=1)               # avoiding Dummy variable Trap
df
df = df.loc[df['price']>10000]              # Otherwise We get a negetive value as our result.
df
x = df.drop(['price','model','transmission','fuelType'],axis=1)
x
y = df['price'].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)
len(x.columns)
x.columns
p = np.where(x.columns == ' X1')[0][0]
print(p)
def pred_price(u_model,u_year,u_transmission,u_mileage,u_fuelType,u_tax,u_mpg,u_enginesize):
  temp_list = np.zeros(len(x.columns))
  temp_list[0] = u_year
  temp_list[1] = u_mileage
  temp_list[2] = u_tax
  temp_list[3] = u_mpg
  temp_list[4] = u_enginesize
  temp_list[np.where(x.columns == u_model)[0][0]] = 1.0
  temp_list[np.where(x.columns == u_transmission)[0][0]] = 1.0
  temp_list[np.where(x.columns == u_fuelType)[0][0]] = 1.0
  print(reg.predict([temp_list]))

u_model = input("Enter the model name: ")
u_model = " "+u_model
u_year = int(input("Enter the year: "))
u_transmission = input("Transmission Type: ").title()
u_mileage = int(input("Enter the mileage: "))
u_fuelType = input("Diesel or Petrol: ").title()
u_tax = int(input("Enter the Tax: "))
u_mpg = float(input("mpg: "))
u_enginesize = float(input("Enter the Engine Size: "))
pred_price(u_model,u_year,u_transmission,u_mileage,u_fuelType,u_tax,u_mpg,u_enginesize)
y_pred = reg.predict(x_test)
type(y_pred)
y_pred = reg.predict(x_test)              
np.set_printoptions(precision=0)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred.reshape(len(y_pred),1)),1))
y_pred
