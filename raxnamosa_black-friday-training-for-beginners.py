# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv ('../input/BlackFriday.csv')
df.head()
df.tail()
df.shape
df.columns
df.info()
df.describe()
df1 = df.drop(["User_ID"], axis=1) # deleted user_ıd
df1.describe()
df1.Product_Category_2.plot(kind = 'line', color = 'black',label = 'Product_Category_2',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')
df1.Product_Category_3.plot(color = 'red',label = 'Product_Category_3',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right') 
plt.xlabel('Product_Category_2')             
plt.ylabel('Product_Category_3')
df1.plot(kind='scatter', x='Product_Category_2', y='Product_Category_3',alpha = 0.5,color = 'red')
plt.xlabel('Product_Category_2')         
plt.ylabel('Product_Category_3')
plt.title('Product_Category_2 and Product_Category_3') 
df1.Product_Category_1.plot(kind='hist',bins=50,figsize=(10,10))
plt.show()
x = df1["Product_Category_1"] < 35
df1[x]
x = df1["Product_Category_1"] > 15
df1[x]
x = df1["Purchase"] < 10000
df1[x]
x = df1["Purchase"] > 10000
df1[x]
trashold = sum(df1.Marital_Status) / len(df1.Marital_Status)
print(trashold)
df1['Marital_Status_level'] = ["high" if i > trashold else "low" for i in df1.Marital_Status]
df1.loc[:13,["Marital_Status_level", "Marital_Status"]]
df1['Product_Category_2'].dropna(inplace = True) # we deleted all NaN Product_Category_2
df1['Product_Category_3'].dropna(inplace = True) # we deleted all NaN Product_Category_3
df1.Product_Category_2[:7] # if it worked, we can check like that.. first 25 columns
df1.Product_Category_3[:7] # and here too
df1.dropna()
df1.loc[1:41, "Gender" : "Stay_In_Current_City_Years"]  # we looked first 15 items of gender and Stay_In_Current_City_Years
df1.loc[13:1:-1, "Gender" : "Age"]   # we looked reverse slicing
df1['Age'] = df1['Age'].astype('category')   # we changed from objeect to category
df1.dtypes
df1.groupby ("Occupation").mean()
df1.groupby ("Occupation")[["Gender", "Stay_In_Current_City_Years"]].max()
df1["Product_Category_3"].value_counts(dropna =False)
df1['Product_Category_3'].dropna(inplace = True) 
assert 1==1
df1["Product_Category_3"]
df1.groupby ("Occupation")[["Product_Category_1", "Product_Category_2", "Product_Category_3"]].min()
