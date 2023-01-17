import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#data loading and view of data

import pandas as pd

file = '/kaggle/input/big-mart-sales/train_v9rqX0R.csv'

df = pd.read_csv(file)

df.head()
print(df.shape) #dimension of data: 8523 rows, 12 columns

print(df.dtypes) #datatypes: measures(4) and categorical variables

print(df["Item_Type"].unique()) #to get unique entries

print(len(df["Item_Type"].unique())) #16 categories of products available

print(len(df["Item_Identifier"].unique())) #1559 unique products types

print(len(df["Outlet_Identifier"].unique())) #10 outlets

print(df["Outlet_Establishment_Year"].unique()) # 1985-2009

print(len(df["Outlet_Establishment_Year"].unique()))
#to find null values in data

df.isna().sum() #found in size of store no: 10,17,45 and item weight in store no:19,27
df['Item_Weight'].fillna(min(df["Item_Weight"]), inplace = True)  

df['Outlet_Size'].fillna("NA", inplace = True)  

df['Item_Fat_Content'].replace(to_replace="LF", value="Low Fat",inplace = True)

df['Item_Fat_Content'].replace(to_replace="low fat", value="Low Fat",inplace = True)

df['Item_Fat_Content'].replace(to_replace="reg", value="Regular",inplace = True)



#print(df.iloc[7])

#print(df.iloc[3])

#print(df.iloc[65])

#print(df.iloc[81])
df.dropna(inplace=True) #deletes row which has atleast 1 na

print(df.shape)
#checking null presence after making changes

df.isna().sum()
#pip install seaborn

import matplotlib.pyplot as plt

import seaborn as sns
#types of store

#sns.catplot(y=df["Outlet_Identifier"],x=df["Outlet_Type"], palette="pastel", data=df)

sns.countplot(y=df["Outlet_Type"], data=df)
sns.countplot(y=df["Outlet_Identifier"], data=df)
#distribution of store across location

#sns.stripplot(y=df["Outlet_Identifier"],x=df["Outlet_Location_Type"], palette="pastel", data=df)

sns.countplot(y=df["Outlet_Location_Type"], data=df)
#Store types,size,location,establishment year, outlet no, sales



year_sale_dx = df.groupby(by=['Outlet_Type', 'Outlet_Identifier','Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Size'])['Item_Outlet_Sales'].sum().reset_index()

#year_sale = year_sale_dx.groupby(by=['Item_Type'])['Item_Outlet_Sales'].transform(min) == year_sale_dx['Item_Outlet_Sales']

#year_sale_min = year_sale_dx[year_sale].reset_index(drop=True)

year_sale_dx.sort_values(by='Item_Outlet_Sales')
#Analysis of sales of item types in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type', 'Item_Type'])['Item_Outlet_Sales'].sum().reset_index()

year_sale = year_sale_dx.groupby(by=['Item_Type'])['Item_Outlet_Sales'].transform(min) == year_sale_dx['Item_Outlet_Sales']

year_sale_min = year_sale_dx[year_sale].reset_index(drop=True)

year_sale_min.sort_values(by='Item_Outlet_Sales')
#count of product items of each category in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type','Item_Type'])['Item_Identifier'].count().reset_index()

year_sale_dx.head(17)
#count of product items of each category in Outlet 27 (comparison with highest revenue store)

year_sale_dx = df.groupby(by=['Outlet_Type','Item_Type'])['Item_Identifier'].count().reset_index()

year_sale_dx.tail(17)
#Visibility of item types in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type','Outlet_Identifier','Item_Type'])['Item_Visibility'].mean().reset_index()

year_sale_dx.head(33)
#Visibility of item types in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type','Item_Type'])['Item_Visibility'].mean().reset_index()

year_sale_dx.tail(17)
#Correlation analysis between variables

df['Item_Visibility'].replace(to_replace="NA", value=min(df["Item_Visibility"]),inplace = True)

corr = df.corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0)
#Min mean Price Comparison between stores to study affect on quantity sales

year_sale_dx = df.groupby(by=['Outlet_Type', 'Item_Type'])['Item_MRP'].mean().reset_index()

year_sale = year_sale_dx.groupby(by=['Item_Type'])['Item_MRP'].transform(min) == year_sale_dx['Item_MRP']

year_sale_min = year_sale_dx[year_sale].reset_index(drop=True)

year_sale_min.sort_values(by='Item_MRP')
#mean price of products under each item category in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type','Item_Type'])['Item_MRP'].mean().reset_index()

year_sale_dx.head(17)
#mean price of products under each item category in Grocery store

year_sale_dx = df.groupby(by=['Outlet_Type','Item_Type'])['Item_MRP'].mean().reset_index()

year_sale_dx.tail(17)
#analysis of count and sales under each product category across stores

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

sns.countplot(ax=axes[0],y=df["Item_Type"],palette="pastel", data=df)

sns.barplot(ax=axes[1],y=df["Item_Type"],x=df["Item_Outlet_Sales"], palette="pastel", data=df)

sns.barplot(ax=axes[2],y=df["Item_Type"],x=df["Item_MRP"], palette="pastel", data=df)
#sales of fat content products

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

sns.countplot(ax=axes[0],x=df["Item_Fat_Content"],palette="pastel", data=df)

sns.barplot(ax=axes[1],x=df["Item_Fat_Content"],y=df["Item_Outlet_Sales"], palette="pastel", data=df)

sns.barplot(ax=axes[2],x=df["Item_Fat_Content"],y=df["Item_MRP"], palette="pastel", data=df)
#products having highest revenue across all stores

cg=df.groupby('Item_Identifier')

z=cg['Item_Outlet_Sales'].agg(['mean','sum']).sort_values(by='mean', ascending=False)

z.head(10)
#under which item type does these products fall under: 



type=['FDR45','NCL42','FDU55','FDZ50','DRK23','FDF39','FDD44','FDT16','FDY55','FDG17']

filt=df['Item_Identifier'].isin(type)

df.loc[filt,['Item_Type','Item_Identifier']]
#Top 10 product items showing in which store they have highest revenue



year_sale_dx = df.groupby(by=['Outlet_Identifier','Item_Type','Item_Identifier'])['Item_Outlet_Sales'].sum().reset_index()

z=year_sale_dx.sort_values(by='Item_Outlet_Sales',ascending=False)

z.head(10)
#Bottomost 10 products having lowest revenue across all stores

cg=df.groupby('Item_Identifier')

z=cg['Item_Outlet_Sales'].agg(['mean','sum']).sort_values(by='mean')

z.head(10)
#under which item type does these products fall under: 



type=['NCR42','FDQ60','FDQ04','FDX10','NCN29','NCL41','NCO06','FDF38','FDT02','FDX38']

filt=df['Item_Identifier'].isin(type)

df.loc[filt,['Item_Type','Item_Identifier']]
#Bottomost 10 product items showing in which store they have lowest revenue



year_sale_dx = df.groupby(by=['Outlet_Identifier','Item_Type','Item_Identifier'])['Item_Outlet_Sales'].sum().reset_index()

z=year_sale_dx.sort_values(by='Item_Outlet_Sales')

z.head(10)
 #If you like it please do click UPVOTE button