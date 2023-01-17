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
# Import Packages
from decimal import Decimal
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
# Read Database file using pd.read_csv command, pass CSV file name and path
DataSet = pd.read_csv('../input/Warehouse_and_Retail_Sales.csv').fillna(0)
# To show Total Number of Record in Dataset
TotalRowCount = len(DataSet)
print("Total Number of Data Count :", TotalRowCount)
# Print Header and First 10 Rows from Dataset
DataSet.head(10)
# Print Header and First 10 Rows from Dataset
DataSet.tail(10)
# Rename Dataset Coloumns's Header
DataSet.rename(columns={'YEAR' : 'Year',
                        'MONTH' : 'Month',
                        'SUPPLIER' : 'Supplier',
                        'ITEM CODE' : 'Item_Code',
                        'ITEM DESCRIPTION' : 'Item_Description',
                        'ITEM TYPE' : 'Item_Type',
                        'RETAIL SALES' : 'Retail_Sales',
                        'RETAIL TRANSFERS' : 'Retail_Transfers',
                        'WAREHOUSE SALES' : 'Warehouse_Sales'}, inplace=True)
# Print again Head to verify Coloumns's Header changed
DataSet.head(10)
# Item wise count
DataSet = DataSet.dropna()
ItemCount = DataSet["Item_Type"].value_counts().nlargest(10)
print("Item Wise Count \n")
print(ItemCount)
sn.set_context("talk",font_scale=1)
plt.figure(figsize=(22,5))
sn.countplot(DataSet['Item_Type'],order = DataSet['Item_Type'].value_counts().index)
plt.title('Item Wise Count \n')
plt.ylabel('Total Count')
plt.xlabel('Item Type')
# Draw Pie Chart
plt.figure(figsize=(6,5))
DataSet['Item_Type'].value_counts().nlargest(5).plot(kind='pie')
plt.title('Top 5 Items \n')
plt.ylabel('Item')
# Find Unique Field Value
UniqueItem = DataSet['Item_Type'].unique()
print("All Unique Items \n")
print(UniqueItem)
# Item Wise Summary. 
# We will take Wine Iteam as example
ItemData=DataSet[DataSet['Item_Type']=='WINE']
print ("The Max Wine Retail Transfers is :",ItemData['Retail_Transfers'].max())
print ("The Min Wine Retail Transfers is :",ItemData['Retail_Transfers'].min())
ItemTypeMean = ItemData['Retail_Transfers'].mean()
print ("The Mean Wine Retail Transfers is :", round(ItemTypeMean,2))
# Top 5 Items in Bar Chart
plt.figure(figsize=(15,5))
GraphData=DataSet.groupby('Item_Type').size().nlargest(5)
GraphData.plot(kind='bar')
plt.title('Top 5 Items \n')
plt.ylabel('Total Count')
plt.xlabel('Item Type')
# Top 5 Supplieies in Bar Chart
plt.figure(figsize=(15,5))
TopFiveSupplier=DataSet.groupby('Supplier').size().nlargest(5)
print(TopFiveSupplier)
TopFiveSupplier.plot(kind='bar')
plt.title('Top 5 Suppliers \n')
plt.ylabel('Total Count')
plt.xlabel('Supplier Name')
# Top 10 Wine Supplieies in Bar Chart
Item=DataSet[DataSet['Item_Type']=='WINE']
Item[Item["Item_Type"]=='WINE']['Supplier'].value_counts()[0:10].to_frame().plot.bar(figsize=(15,5))
ItemSupplier = Item[Item["Item_Type"]=='WINE']['Supplier'].value_counts()[0:10]
print("Top 10 Wine Suppliers \n")
print(ItemSupplier)
plt.title('Top 10 Wine Suppliers\n')
plt.ylabel('Wine Count')
plt.xlabel('Supplier Name')
# Conclusion
# Total Number of Data : 128355
# Top Supplier : REPUBLIC NATIONAL DISTRIBUTING CO
# Top Item : Wine
# Top Wine Suppier : A VINTNERS SELECTIONS