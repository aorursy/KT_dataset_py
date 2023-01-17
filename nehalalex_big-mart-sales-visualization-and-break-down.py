import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.ticker as tick
data = pd.read_csv("../input/big-mart-sales/bigmartsales.csv")

data.head()  #first few rows 

data.tail()#Last few rows
data.shape
data.info()
print(data.columns)
data.describe() #shows basic statistical characteristics of each numerical feature (int64 and float64 types)
data.describe(include=['object', 'float']) 
data['Item_Fat_Content'].unique()
data['Item_Type'].unique()
print(data.isna().sum())
print('Categorical Value Count for the following columns:\n')

print('Outlet_Size:')

print(data['Outlet_Size'].value_counts())

print('\nOutlet_Location_Type:')

print(data['Outlet_Location_Type'].value_counts())
#Replace the following unintelligible categories with np.nan

data.replace({"?": np.nan, "  --": np.nan, "na": np.nan, "NAN": np.nan, "  -": np.nan}, inplace = True)

print(data.isna().sum())



# Filling missing values of Item_Weight

data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)#filling null values with mean value
data['Outlet_Location_Type'].fillna(method="ffill",inplace = True)

data['Outlet_Size'].fillna(method="ffill",inplace = True)

data
replace={'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'}

data.Item_Fat_Content.replace(replace, inplace=True)

print(data.Item_Fat_Content.value_counts())
data.isna().sum()
data.loc[0:3, 'Item_Identifier':'Item_Type']
data.iloc[0:3, 0:13]
data[-1:]
import numpy as np

data.apply(np.max) 
plt.rcParams['figure.figsize']=(5,5)

plt.bar(['Low Fat','Regular'],data.Item_Fat_Content.value_counts(),width=0.5,color=['blue', 'cyan'],edgecolor='yellow')

plt.xlabel('Item_Fat_Content')

plt.ylabel('Count')

plt.title('Count of Item_Fat_Content')

plt.show()
plt.figure(figsize=(25,7))

sns.countplot('Item_Type',data=data,palette='spring')

plt.xlabel('Item_Type')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(8,5))

sns.countplot('Outlet_Size',data=data,palette='Purples')

plt.xlabel('Outlet_Size')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(8,5))

sns.countplot('Outlet_Type',data=data,palette='autumn')

plt.xlabel('Outlet_Type')

plt.ylabel('Count')

plt.show()
Outlet_Type_pivot = data.pivot_table(index='Item_Fat_Content', values="Item_Outlet_Sales", aggfunc=np.sum)

Outlet_Type_pivot.plot(kind='bar', color='blue',figsize=(5,5))

plt.xlabel("Item_Fat_Content ")

plt.ylabel("Item_Outlet_Sales")

plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")

plt.xticks(rotation=0)

plt.ticklabel_format(axis="y", style="plain")

plt.show()
df3=data.groupby(by='Item_Type').sum()

df2=df3['Item_Outlet_Sales'].sort_values(ascending=False)

plt.rcParams['font.size'] = 10

plt.pie(df2, autopct = '%0.1f%%', radius = 2.0, labels = ['Fruits and Vegetables', 'Snack Foods','Household ','Frozen Foods','Dairy ', 'Canned','Baking Goods','Health and Hygiene','Meat', 'Soft Drinks','Breads','Hard Drinks','Starchy Foods', 'Others','Breakfast','Seafood'],

      explode = [0.2,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0.0,0.0],colors=['#ff6666', '#ffcc99', '#99ff99', '#66b3ff'])

plt.show()

type1=data.groupby(['Outlet_Identifier'])['Item_Outlet_Sales'].sum()
plt.rcParams['figure.figsize']=(10,6)

a=['OUT010','OUT013','OUT017','OUT018','OUT019','OUT027','OUT035','OUT045','OUT046','OUT049']

plt.bar(a,type1,color='gold',width=0.6)

plt.xlabel('Outlet_Store_ID')

plt.ylabel('Sales')

plt.title('Outlet vs Sales')

plt.show()
plt.figure(figsize=(10,5))

type2=data.groupby(['Outlet_Type'])['Item_Outlet_Sales'].sum()

store_types=['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']

plt.bar(store_types,type2, width=0.6,color='coral')

plt.ticklabel_format(axis="y", style="plain")

plt.xlabel('Outlet_Type')

plt.ylabel('Item_Outlet_Sales')

plt.title('Outlet Type vs Outlet Sales')

plt.show()
#sci to plain

#x and y

plt.figure(figsize=(10,5))

type3 = data.groupby(['Outlet_Size'])['Item_Outlet_Sales'].sum()

size = ['High', 'Medium', 'Small']

plt.bar(size, type3, color='Orange',width=0.6)

plt.ticklabel_format(axis="y", style="plain")

plt.xlabel('Outlet_Size')

plt.ylabel('Item_Outlet_Sales')

plt.title('Outlet Type vs Outlet Sales')

plt.show()
plt.rcParams['figure.figsize'] = 25,5

chart=sns.boxplot(x="Item_Type",y="Item_MRP",data=data,palette='husl')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45,horizontalalignment='right', fontweight='light',fontsize='x-large')

plt.show()
plt.figure(figsize=(10,5))

sns.boxplot('Outlet_Establishment_Year','Item_Outlet_Sales',data=data,palette="Paired")

plt.show()
data["Total_Profit"] = (data['Item_Outlet_Sales']/data['Item_MRP']) * data['Profit']

type6=data.groupby(by='Item_Type')['Total_Profit'].sum()

plt.figure(figsize = (10,5))

lab = ['Fruits and Vegetables', 'Snack Foods','Household ','Frozen Foods','Dairy ', 'Canned','Baking Goods','Health and Hygiene','Meat', 'Soft Drinks','Breads','Hard Drinks','Starchy Foods', 'Others','Breakfast','Seafood']

lab.sort()

chart = sns.barplot(x = lab, y = type6)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light',fontsize='x-large')

plt.xlabel('Total_Profit')

plt.ylabel('Item_Outlet_Sales')

plt.title('Profit vs Item_Type')

plt.show()