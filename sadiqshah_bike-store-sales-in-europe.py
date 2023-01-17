import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



Sales=pd.read_csv('/kaggle/input/bike-sales-in-europe/Sales.csv')
Sales.head()
Sales['Country'].unique()
Shape = Sales.shape

Rows=Shape[0]

Col= Shape[1]

print(f"Rows of DataSet is :  {Rows}")

print(f"Columns of DataSet is :  {Col}")
Sales.info()
Sales['Date'] = pd.to_datetime(Sales['Date'])
Sales["Calculated_Date"]= Sales[['Year', 'Month', 'Day']].apply(lambda x: '{}-{}-{}'.format(x[0], x[1], x[2]), axis=1)

Sales["Calculated_Date"]=pd.to_datetime(Sales['Calculated_Date'])

Sales["Calculated_Date"].head()

Sales.isnull().sum()
Sales.describe()
Sales['Unit_Cost'].describe()

Sales['Unit_Cost'].mean()
Sales['Unit_Cost'].median()
Sales['Unit_Cost'].plot(kind="box",figsize=(14,6),vert=False, fontsize=12)
Sales['Unit_Cost'].plot(kind="density" , figsize=(12,6), fontsize=12)

plt.title("Unit Cost",fontsize=15)

plt.xlabel("Unit Cost",fontsize=12)

plt.ylabel("Sales",fontsize=12)
ax=Sales['Unit_Cost'].plot(kind="density" , figsize=(12,6))

# ax.axvlines(Sales['Unit_Cost'].mean())

mean= ax.axvline(Sales['Unit_Cost'].mean(), color='red' )

median = ax.axvline(Sales['Unit_Cost'].median(), color='g')

plt.legend({'Median':median,'Mean':mean})
ax = Sales['Unit_Cost'].plot(kind='hist', figsize=(14,6))

ax.set_ylabel('Number of Sales', fontsize=15)

ax.set_xlabel('Dollars', fontsize=15)
Sales["Customer_Age"].value_counts().mean()
Sales["Customer_Age"].plot(kind='box',vert=False,figsize=(15,6), fontsize=12)
Sales["Customer_Age"].plot(kind='kde',figsize=(12,6))

plt.title("Sales",fontsize=15)

plt.xlabel("Customer Age",fontsize=12)

plt.ylabel("Sales",fontsize=12)

plt.legend()
Sales['Year'].value_counts()
Sales["Year"].value_counts().plot(kind="bar",figsize=(16,6))
Sales['Month'].value_counts()
Sales["Month"].value_counts().plot(kind="bar",figsize=(12,6))

plt.legend()
Sales['Calculated_Date'].value_counts().plot(kind="line",figsize=(14,6))

plt.legend()
Sales['Revenue']+50
Sales['Age_Group'].value_counts()
Sales["Age_Group"].value_counts().plot(kind='bar',figsize=(14,6))

plt.legend()

plt.legend()

plt.ylabel("Sales")

plt.xlabel("Age")
Sales["Age_Group"].value_counts().plot(kind='pie',figsize=(14,8),autopct='%1.1f%%',fontsize=13)

plt.title("Age Group",fontsize=14)
Sales["Order_Quantity"].mean()
Sales["Order_Quantity"].plot(kind='box',vert=False,figsize=(12,6),fontsize=12)

plt.title("Order Quality",fontsize=15)

Sales['Country'].value_counts()
Sales['Country'].value_counts().plot(kind='bar',figsize=(12,4))

plt.title("Sales In Each Country",fontsize=17)

plt.ylabel("Sales",fontsize=15)

plt.xlabel("Country",fontsize=15)

plt.legend()
MostSales=Sales.loc[: ,"Product"].unique()

# sales['Product'].unique()
Sales.loc[: ,"Product"].value_counts().head(10).plot(kind='bar',figsize=(14,5))

plt.title("List of Products Sales",fontsize=17)

plt.ylabel("Number of Products",fontsize=15)

plt.xlabel("Products",fontsize=15)

plt.legend()
Sales.plot(kind='scatter',x="Unit_Cost", y="Unit_Price",figsize=(12,4),fontsize=10)

plt.title("Relation between Unit cost & Unit Price",fontsize=17)

plt.ylabel("Unit Price",fontsize=15)

plt.xlabel("Unit Cost",fontsize=15)

Sales.plot(kind="box",x='Order_Quantity',y='Profit',figsize=(12,4),fontsize=10)

plt.title("Relation Between Order Quantity & Profit",fontsize=15)

Sales.plot(kind="box",x='Country',y='Profit',figsize=(12,4),fontsize=13,vert=False )

plt.title("Relation between Country & Unit Profit",fontsize=17)

Sales[["Customer_Age","Country"]].boxplot(by='Country',figsize=(10,6))
Sales.loc[((Sales['Country']=='Canada' ) |  (Sales['Country']=='France' ))].shape[0]
Sales.loc[(Sales['Country']=='Canada' ) & (Sales['Sub_Category']=="Bike Racks")].shape[0]
Sales.loc[Sales["Country"]=="France","State"].value_counts()
Sales.loc[Sales["Country"]=="France","State"].value_counts().plot(kind='bar',figsize=(16,5))

plt.title("Sales in Each state of France",fontsize=17)

plt.ylabel("Number of Sales",fontsize=15)

plt.xlabel("States",fontsize=15)

plt.legend()



plt.legend()
Sales['Sub_Category'].value_counts()
Sales['Product_Category'].value_counts().plot(kind='pie',figsize=(8,8),autopct='%1.1f%%',fontsize=12)

Cat=Sales['Sub_Category'].unique()

plt.title("Products Category",fontsize=15)

plt.legend(wedges, Cat,fontsize=12,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))



Sales["Sub_Category"].value_counts()
Sales.loc[Sales['Product_Category']=='Accessories' ,"Sub_Category"].value_counts()
Sales.loc[Sales['Product_Category']=='Accessories' ,"Sub_Category"].value_counts().plot(kind="bar",figsize=(16,4))

plt.legend()
Sales.loc[Sales['Product_Category']=="Bikes","Sub_Category"].value_counts()
Pc=Sales.loc[Sales['Product_Category']=="Bikes","Sub_Category"].unique()

Sales.loc[Sales['Product_Category']=="Bikes","Sub_Category"].value_counts().plot(kind="pie",figsize=(8,8),fontsize=12)

plt.legend(fontsize=13,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))

plt.title("Products Category in Bikes",fontsize=15)



Sales['Customer_Gender'].value_counts()
Sales['Customer_Gender'].value_counts().plot(kind='bar',fontsize=12,figsize=(8,6))

plt.ylabel("Sales",fontsize=13)

Sales.loc[(Sales['Customer_Gender']=="M") & (Sales["Revenue"]>=500)].shape[0]
Sales.sort_values(['Revenue'],ascending=False).head(5)
# Sales.sort_values(['Revenue'],ascending=False).head(1)



Sales['Revenue'].max()



# Cond=Sales['Revenue']==Sales["Revenue"].max()

# Sales.loc[Cond]


Sales.loc[Sales["Revenue"]>10_000 , "Order_Quantity"].mean()



# cond = Sales['Revenue'] > 10_000

# Sales.loc[cond, 'Order_Quantity'].mean()

Sales.loc[Sales["Revenue"]<10_000,"Order_Quantity"].mean()
Sales.loc[(Sales["Year"]==2016) & (Sales["Month"]=='May')].shape[0]
Sales.loc[(Sales['Year'] == 2016) & (Sales['Month'].isin(['May', 'June', 'July']))].shape[0]
Saels2016=Sales.loc[Sales["Year"]==2016 , ['Profit',"Month"]]

Saels2016.boxplot(by="Month", figsize=(14,6))
Sales.loc[Sales["Country"]=="United State",'Unit_Price']*=1.072
Sales["Unit_Price"].head(2)
plt.figure(figsize=(16,8))

Sales.plot(kind="scatter",x="Customer_Age",y="Revenue",figsize=(10,8),fontsize=12)

plt.xlabel("Customer Age",fontsize=13)

plt.ylabel("Revenue",fontsize=13)

plt.show()
Sales.plot(kind='scatter', x='Revenue', y='Profit', figsize=(10,8),fontsize=12)

plt.xlabel("Revenue",fontsize=13)

plt.ylabel("Profit",fontsize=13)
Sales["Revenu_Per_Age"] = Sales["Revenue"]/Sales['Customer_Age']
Sales['Revenu_Per_Age'].plot(kind='density', figsize=(14,6))

plt.title("Revenue Per Age",fontsize=15)





plt.legend()
Sales['Revenu_Per_Age'].plot(kind='hist', figsize=(14,6))

plt.title("Revenue Per Age",fontsize=15)

plt.legend()
Sales['Calculated_Cost'] = Sales['Order_Quantity'] * Sales['Unit_Cost']
Sales['Calculates_Revenue']= Sales["Cost"] + Sales["Profit"]
Sales["Revenue"].plot(kind="hist" , bins=100 ,figsize=(14,6))

plt.legend()
Tax = 1.03

Unit_Price_Tax=Sales['Unit_Price']*Tax
Unit_Price_Tax.plot(kind="hist",figsize=(12,4))

plt.xlabel("Unit_Price",fontsize=13)

plt.ylabel("Sales",fontsize=13)

plt.legend()


Sales.loc[Sales["State"]=='Kentucky'].head()
Sales.loc[Sales['Age_Group'] == 'Adults (35-64)', 'Revenue'].mean()
Sales.loc[(Sales['Age_Group'] == 'Youth (<25)') | (Sales['Age_Group'] == 'Adults (35-64)')].shape[0]
Sales.loc[(Sales['Age_Group'] == 'Adults (35-64)') & (Sales['Country'] == 'United States'), 'Revenue'].mean()
Revenue_France=Sales.loc[Sales['Country']=="France",'Revenue']

Revenue_France*=1.1
Revenue_France
Corr=Sales.corr()

Corr
figure = plt.figure(figsize=(8,8))

plt.matshow(Corr, cmap='RdBu', fignum=figure.number)

plt.xticks(range(len(Corr.columns)),Corr.columns,rotation='vertical')

plt.yticks(range(len(Corr.columns)), Corr.columns);