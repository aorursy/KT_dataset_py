import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv('/kaggle/input/bigmart-sales-dataset/EDA_BIGMART_SALES.csv',index_col=0)

data.head()
data.columns
num_var = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['int64','float64']]

cat_var = [key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['object']]
print(num_var)

print(cat_var)
sns.distplot(data.Item_Outlet_Sales)
sns.scatterplot(x=data.Item_MRP,y=data.Item_Outlet_Sales,hue=data.Item_Type_Category)
sns.scatterplot(x=data.Item_Visibility,y=data.Item_Outlet_Sales,hue=data.Item_Type_Category)
plt.bar(x=data.Outlet_Establishment_Year,height=data.Item_Outlet_Sales)
sns.boxplot(x=data.Item_Type_Category,y=data.Item_Outlet_Sales)
plt.figure(figsize=(8,5))

sns.barplot(x="Outlet_Type",data=data,hue='Item_Type_Category',y="Item_Outlet_Sales")

plt.show()
plt.figure(figsize=(8,5))

sns.barplot(x="Item_Fat_Content",data=data,hue="Item_Type_Category",y="Item_Outlet_Sales")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(x="Item_Fat_Content",data=data,hue="Item_Type_Category")

plt.show()
plt.pie(data.Item_Fat_Content.value_counts(),labels=data.Item_Fat_Content.unique(),autopct='%1.1f%%')
plt.pie(data.Item_Type_Category.value_counts(),labels=data.Item_Type_Category.unique(),autopct='%1.1f%%')
plt.pie(data.Outlet_Location_Type.value_counts(),labels=data.Outlet_Location_Type.unique(),autopct='%1.1f%%')