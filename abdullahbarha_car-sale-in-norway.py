# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
car_data=pd.read_csv('../input/norway_new_car_sales_by_month.csv')

car_data.info()
car_data.corr()
car_data.head(10)
car_data.columns
bymakedata=pd.read_csv("../input/norway_new_car_sales_by_make.csv")
bymodeldata=pd.read_csv("../input/norway_new_car_sales_by_model.csv")
bymakedata.head()
car_data.describe()
car_data.Quantity.plot(x='Year', kind='line', color='green', label='Total_Quantity', linewidth=1, alpha=0.7, grid=True, linestyle='-',figsize=(10, 10))
car_data.Quantity_Electric.plot(color='blue', label='Quantity_Electric', linewidth=3, alpha=0.9, grid=True, linestyle=':' )
car_data.Quantity_Diesel.plot(color='red', label='Quantity_Diesel', linewidth=1, alpha=0.9, grid=True, linestyle='-')
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Quantity')

plt.title('Quantity vs Quantity Diesel')
plt.show()
#Diesel_Share - share of diesel cars in total sales (Quantity_Diesel / Quantity)
car_data.Diesel_Share.plot(x='Year', kind='line', color='green', label='Diesel_Share', linewidth=3, alpha=0.7, grid=True, linestyle='-',
                           figsize=(10, 10))
car_data.Diesel_Co2.plot(color='red', label='Diesel_Co2', linewidth=3, alpha=0.9, grid=True, linestyle='-')
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Quantity')

plt.title('Diesel Share vs Diesel_Co2')
plt.show()
car_data.Quantity.plot(x='Year', kind='line', color='green', label='Quantity', linewidth=3, alpha=0.7, grid=True, linestyle='-',
                           figsize=(10, 10))
car_data.Import.plot(color='red', label='Import', linewidth=3, alpha=0.9, grid=True, linestyle='-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
car_data.plot(kind='scatter', x='Quantity_Diesel', y='Diesel_Co2', alpha=0.9, color='red', figsize=(10, 10))
plt.xlabel('Avg_CO2')
plt.ylabel('Quantity_Hybrid')
plt.show()
x=car_data['Quantity']<10000
car_data[x]
car_data.Quantity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# mean total quantity is about 10000
car_data=pd.read_csv('../input/norway_new_car_sales_by_month.csv')
car_data.head()
car_data.shape
car_data.columns
car_data.info()
car_data.columns
print(car_data['Year'].value_counts(dropna=False))
car_data.describe()
car_data.boxplot(column='Avg_CO2',by='Import', figsize = (15,15))
plt.show()
data_new=car_data.head()
data_new.head(10)
melted=pd.melt(frame=data_new,id_vars='Year',value_vars=['Bensin_Co2','Avg_CO2'])
melted
data1=car_data.head()
data2=car_data.tail()
conc_data_row=pd.concat([data1,data2],axis=0, ignore_index=True)
conc_data_row
data3=car_data['Quantity'].head()
data4=car_data['Quantity_Diesel'].head()
conc_data_=pd.concat([data3,data4],axis=1)
conc_data_

car_data.dtypes
# converst all data float to int
car_data['Quantity']=car_data['Quantity'].astype('float')
car_data['Quantity_YoY']=car_data['Quantity_YoY'].astype('float')
car_data['Import']=car_data['Import'].astype('float')
car_data['Import_YoY']=car_data['Import_YoY'].astype('float')
car_data['Avg_CO2']=car_data['Avg_CO2'].astype('float')
car_data['Bensin_Co2']=car_data['Bensin_Co2'].astype('float')
car_data['Diesel_Co2']=car_data['Diesel_Co2'].astype('float')
car_data['Quantity_Diesel']=car_data['Quantity_Diesel'].astype('float')
car_data.dtypes
car_data.info()
car_data['Used'].value_counts(dropna=False)
# NaN values dropped
car_data['Used'].dropna(inplace=True)
assert car_data['Used'].notnull().all()
car_data['Used'].value_counts(dropna=False)
assert car_data['Used'].notnull().all()
car_data["Used"].fillna('empty',inplace=True)
assert car_data['Used'].notnull().all()

car_data['Used'].value_counts(dropna=False)

