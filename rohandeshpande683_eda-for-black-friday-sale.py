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
#looking for the csv files
import glob
pattern = '*.csv'
csv_files = glob.glob(pattern)
print(csv_files)
#importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset using pandas
df = pd.read_csv("../input/BlackFriday.csv")

#checking the dataset with the head method.
df.head()
#checking the bottom lines of the dataset
df.tail(10)
#checking rest other parameters of the dataset
df.shape
df.info()
#info method in pandas showed that in product_category_2 & 3 there are some null values
#this also displays the datatype of all the columns. If required datatype can be changed.
df.describe()
#describe method will give the statistical output for all the numecial columns in the dataset.
#checking the unique counts of the some of the fiels in the dataset
df.Age.unique()
#print(type(df.Age.unique()))
#unique values in User_ID column
df.User_ID.unique()
#Uniqe values in the product_category column 1
df.Product_Category_1.unique()
#Unique values in the product_category column 2
df.Product_Category_2.unique()
#Unique values in the product_category column 3
df.Product_Category_3.unique()
#Unique values in occupation
df.Occupation.unique()
#now checking the unique values against the key columns such as occupation
print(df['Occupation'].value_counts(dropna=False))
#it appears that occupation with the id 4, 0, 7, 1 & 17 have done max shopping.
#check gender wise shopping. withe the parameter dropna = false, we are instructing system
#not to could null values
print(df['Gender'].value_counts(dropna=False))
print(df.Gender.value_counts(normalize=True))
#from the below data it appears that 75% shopping has been done by males
#checking the marital_status
print(df['Marital_Status'].value_counts(dropna=False))
print(df.Marital_Status.value_counts(normalize=True))
#now checking the dataset based on Age
print(df['Age'].value_counts(dropna=False))
print(df.Age.value_counts(normalize=True))
#Highest shoppers are between age-group of 26-35
#checkin the data further more
print('Min Purchase is of', df.Purchase.min())
print('Max Purchase is of', df.Purchase.max())
print('Average Purchase value is', df.Purchase.mean())
print('Median Purchase value is', df.Purchase.median())
#checking the product category & city_category
#print(df['Product_Category_1'].value_counts(dropna=False))
print(df.Product_Category_1.value_counts(normalize=True))
print(df.Product_Category_2.value_counts(normalize=True))
print(df.Product_Category_3.value_counts(normalize=True))
print(df.City_Category.value_counts(normalize=True))
#Changing the datatype of the few items to category.
#df.info()
df_User_ID = df.User_ID.astype('category')
df_Product_ID = df.Product_ID.astype('category')
df_Gender = df.Gender.astype('category')
df_Age = df.Age.astype('category')
df_City_Category = df.City_Category.astype('category')
df_Marital_Status = df.Marital_Status.astype('category')
df_Product_Category_1 = df.Product_Category_1.astype('category')
df_Product_Category_2 = df.Product_Category_2.astype('category')
df_Product_Category_3 = df.Product_Category_3.astype('category')
print(df.info())
#recoding Gender for pratice
def recode_gender(Gender):
    if Gender == 'M':
        return 1
    elif Gender =='F':
        return 2
    else:
        return np.nan
df['recode'] = df.Gender.apply(recode_gender)
print(df.head())

# As checked in the earlier, when the information is displayed there are some null 
#values in the Product_Category_2 & 3 columns. Hence filling those values will nan.
df['Product_Category_2'] = df.Product_Category_2.fillna('nan')
df['Product_Category_3'] = df.Product_Category_3.fillna('nan')
print(df.info())
#visually inspecting and plotting the data with numpy library
Age = np.array(df['Age'])
Marital_Status = np.array(df['Marital_Status'])
Purchase = np.array(df['Purchase'])

#plotting this on line graph

plt.scatter(Age, Purchase)
#plt.show()

plt.scatter(Marital_Status, Purchase)
plt.show()
#plotting using pandas and sinlge variable
df['Purchase'].plot(kind='hist', rot = 100, logx=True, logy=True, bins= 30)
plt.show()
#Checking purchases by age with the help of boxplot
df.boxplot(column = 'Purchase', by='Age')
plt.show()
#checking the visualization with scatter plot
df.plot(kind='scatter', x='Occupation', y='Purchase', rot=70)
plt.show()
#plotting with pandas
#y_columns = ['Occupation']
df.plot(x='Purchase', y= 'Occupation')
plt.show()
#plotting Probability density function for the purchase coloumn
fig, axes = plt.subplots(nrows=2, ncols=1)
df.Purchase.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(1000,30000))
plt.show()

#now plotting cumulative density functions for the purchase coloumn in the dataset
fig, axes = plt.subplots(nrows=2, ncols=1)
df.Purchase.plot(ax=axes[1],kind='hist', normed=True, cumulative=True,bins=30,range=(1000,30000))
plt.show()
#plotting the min and the max value from the purchase coloumn
print(df.Purchase.min())
print(df.Purchase.max())

mean = df.mean(axis='columns')
mean.plot()
plt.show()
#printing statistics of the Purchase column
print(df.Purchase.describe())
df.Purchase.plot(kind='box')
plt.show()