# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
Data=pd.read_csv("../input/BlackFriday.csv")

# Any results you write to the current directory are saved as output.
Data.head(n=10)
Data.info()
#So there are 5891 Customers
len(Data["User_ID"].unique())
#So there are 3623 Customers
len(Data["Product_ID"].unique())
#Lets talk about Gender of all customers
Data_G=Data.loc[:,["User_ID","Gender"]]
Data_G=Data_G.drop_duplicates()
Data_G.head()
#So there are more male customer than female
#I thought female are more interested in shoppinnnng but i am wrong.
sns.countplot(data=Data_G, x = 'Gender')
plt.show()
# Some piechart for Gender data
D_G=pd.DataFrame(Data_G["Gender"].value_counts())
D_G.plot(kind='pie', subplots=True, figsize=(7, 7),explode=(0.1,0),autopct='%1.1f%%')
plt.show()
#Let's talk about age factor
Data_A=Data.loc[:,["User_ID","Age"]]
Data_A=Data_A.drop_duplicates()

#Total number of unique customers based on age data
Data_A["Age"].value_counts()
# So people having Age group 26-35 are more interested in shopping on fridays
D_A=pd.DataFrame(Data_A["Age"].value_counts())
sns.countplot(data=Data_A, x = 'Age')
plt.show()
#Some occupation Data
#So Consumer with Occupation Id 4 have done most transactions
Data_Occup=Data.loc[:,["User_ID","Occupation"]]
Data_Occup=Data_Occup.drop_duplicates()
D_O=pd.DataFrame(Data_Occup["Occupation"].value_counts())
sns.countplot(data=Data_Occup, x = 'Occupation')
plt.show()
#ANalysis on City Data
Data_City=Data.loc[:,["User_ID","City_Category"]]
Data_City=Data_City.drop_duplicates()
sns.countplot(data=Data_City, x = 'City_Category')
plt.show()
D_C=pd.DataFrame(Data_City["City_Category"].value_counts())
D_C.plot(kind="pie",subplots=True, figsize=(5, 5),autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(Data['City_Category'],hue=Data['Age'])
plt.show()
#Let's talk about total purchase
#Let's see which user has done maximum amout of sale
Total_Purchase=Data.loc[:,["User_ID","Purchase"]]
z=Total_Purchase.groupby("User_ID")
x=[]
for i,j in z:
    x.append([i,np.sum(j["Purchase"])])
Data_g_x=pd.DataFrame(x,columns=["User_ID","Total_Purchase"])
Data_g_x["Total_Purchase"].head()
l=Data_g_x.nlargest(10,"Total_Purchase")
l["User_ID"]=l["User_ID"].astype(str)
plt.figure(figsize=(10,5))
sns.barplot(l["User_ID"],l["Total_Purchase"])
plt.show()
#Let's see  product sales using Product Id
Product_ID=Data["Product_ID"].unique()
#Something about product Category1
#Top 10 most sold Category 1 product
Data_Product1=Data.loc[:,["Product_ID","Product_Category_1"]]
Data_Product1=Data_Product1.groupby("Product_ID")
z=[]
for i,j in Data_Product1:
    z.append([i,np.sum(j["Product_Category_1"])])
Data_Product1=pd.DataFrame(z,columns=["Product_ID","Total_products"])
Data_Product1=Data_Product1.nlargest(10,"Total_products")
Data_Product1["Product_ID"]=Data_Product1["Product_ID"].astype(str)
plt.figure(figsize=(15,5))
sns.barplot(Data_Product1["Product_ID"],Data_Product1["Total_products"])
plt.show()
#Something about product Category2
#Top 10 most sold Category 2 product
Data_Product2=Data.loc[:,["Product_ID","Product_Category_2"]]
Data_Product2=Data_Product2.groupby("Product_ID")
z=[]
for i,j in Data_Product2:
    z.append([i,np.sum(j["Product_Category_2"])])
Data_Product2=pd.DataFrame(z,columns=["Product_ID","Total_products"])
Data_Product2=Data_Product2.nlargest(10,"Total_products")
Data_Product2["Product_ID"]=Data_Product2["Product_ID"].astype(str)
plt.figure(figsize=(15,5))
sns.barplot(Data_Product1["Product_ID"],Data_Product1["Total_products"])
plt.show()
#Something about product Category3
#Top 10 most sold Category 3 product

Data_Product3=Data.loc[:,["Product_ID","Product_Category_3"]]
Data_Product3=Data_Product3.groupby("Product_ID")
z=[]
for i,j in Data_Product3:
    z.append([i,np.sum(j["Product_Category_3"])])
Data_Product3=pd.DataFrame(z,columns=["Product_ID","Total_products"])
Data_Product3=Data_Product3.nlargest(10,"Total_products")
Data_Product3["Product_ID"]=Data_Product3["Product_ID"].astype(str)
plt.figure(figsize=(15,5))
sns.barplot(Data_Product3["Product_ID"],Data_Product3["Total_products"])
plt.show()
#Let's see which product category has been mostly sold on the basis of quantity 
Data_X=Data.loc[:,["Age","Product_Category_1","Product_Category_2","Product_Category_3"]]
Data_X
col=list(Data_X.columns[1:])
Data_X["Product_Category_1"].sum()
sns.barplot(col,[Data_X["Product_Category_1"].sum(),Data_X["Product_Category_2"].sum(),Data_X["Product_Category_3"].sum()])
plt.show()