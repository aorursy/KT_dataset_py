import os
print(os.listdir("../input"))
import os
os.getcwd()
os.chdir()
from IPython.display import Image
Image(filename ="../input/data-dictionary.png",width= 700, height= 700 )

import pandas as pd
import numpy as np
mart_train = pd.read_csv('../input/Train.csv')
mart_test = pd.read_csv('../input/Test.csv')
mart_data = pd.concat([mart_train, mart_test],ignore_index = True, axis = 0)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
mart_data.shape
mart_train.shape
mart_test.shape
mart_train.head()
mart_test.head()
mart_data.apply(lambda x : sum(x.isnull()))
#First confirming Null values of Item_Sales
mart_train.apply(lambda x : sum(x.isnull()))
mart_data.info()
# Looking at how many Levels exist in Categorical variables and their Significance.
mart_data.apply(lambda x : len(x.unique()))
mart_data.Item_Fat_Content.unique()
mart_data['Item_Fat_Content'] = mart_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'])
mart_data.Item_Fat_Content.unique()
mart_data['Item_Type'].unique()
# Looking at the distribution of data
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize = (8,4))
mart_data['Item_Type'].value_counts().plot('bar')
plt.title('Distribution of Item_type across categories', fontsize = 20)
plt.ylabel('No of Items', fontsize = 15)
plt.show();
def substring(ItemId):
    Id = ItemId[:2]
    return(Id)
mart_data.ID = mart_data.Item_Identifier.apply(substring)
mart_data.ID.unique()
# finding the Mappings between Item Id and Type
mart_data.Item_Type[mart_data.ID == 'FD'].unique()
mart_data.Item_Type[mart_data.ID == 'DR'].unique()
mart_data.Item_Type[mart_data.ID == 'NC'].unique()
mart_data.Item_Type = mart_data.Item_Type.replace(to_replace = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Baking Goods',
       'Snack Foods', 'Frozen Foods', 'Breakfast', 'Canned', 'Breads',
       'Starchy Foods', 'Seafood','Soft Drinks', 'Hard Drinks', 'Dairy','Household', 'Health and Hygiene', 'Others'],
                            value = ['FD','FD','FD','FD','FD','FD','FD','FD','FD','FD','FD','DR','DR','DR','NC','NC','NC'])
mart_data.Item_Type = mart_data.Item_Type.replace(to_replace =['FD','DR','NC'] , value = ['Food','Drinks','Non-Consummable'])
# Visualizing the distribution of data after the change
mart_data.Item_Type.value_counts().plot('bar')
plt.title('Distribution of Item across various Item_categories', fontsize = 15)
plt.ylabel('Number of records')
plt.xticks(rotation = 0)
plt.show();
mart_data.Outlet_Type.unique()
mart_data['Outlet_Size'].unique()
mart_data.Outlet_Type[mart_data['Outlet_Size'].isnull()].count()
mart_data.Outlet_Type[mart_data['Outlet_Size'].isnull()].unique()
mart_data.Outlet_Size[mart_data.Outlet_Type == 'Grocery Store'].unique()
# Impute missing values corresponding to Outlet_Type as Grocery Store as "SMALL"
mart_data[mart_data.Outlet_Type == 'Grocery Store']['Outlet_Size'] = 'Small'
mart_data.Outlet_Size[mart_data.Outlet_Type == 'Grocery Store'].unique()
mart_data[mart_data.Outlet_Size.isnull()]['Outlet_Identifier'].unique()
mart_data[mart_data.Outlet_Size.isnull()]['Outlet_Location_Type'].unique()
mart_data[mart_data.Outlet_Size.isnull()]['Outlet_Type'].unique()
mart_data.Outlet_Size[(mart_data.Outlet_Identifier == 'OUT045') | (mart_data.Outlet_Identifier == 'OUT017')].unique()
mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 2'].unique()
mart_data.Outlet_Size[mart_data.Outlet_Type == 'Supermarket Type1'].unique()
mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 2'].value_counts()
mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 2'].isnull().sum()
mart_data.Outlet_Size[mart_data.Outlet_Type == 'Supermarket Type1'].value_counts()
mart_data['Outlet_Size'].isnull().sum()
#looking at the distribution of Outlet_size across records
mart_data.Outlet_Size.value_counts().plot('bar')
plt.xticks(rotation = 0)
plt.show();
mart_data.pivot_table(index= 'Outlet_Type',values='Item_Outlet_Sales',aggfunc='mean')
#mart_data.Outlet_Size.isnull().sum()
mart_train[mart_train.Outlet_Size.isnull()]['Item_Outlet_Sales'].mean()
mart_data.Outlet_Size[(mart_data.Outlet_Location_Type == 'Tier 2')].value_counts()
len(mart_data.Outlet_Size[(mart_data.Outlet_Location_Type == 'Tier 2')])
mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 2'] = 'Small'
print('Number of missing values after impuation: ',mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 2'].isnull().sum())
mart_data.Outlet_Location_Type[mart_data.Outlet_Size.isnull()].unique()
mart_data.Outlet_Size[mart_data.Outlet_Location_Type == 'Tier 3'].value_counts()
# Imputing missing values to medium outlet_size
mart_data.Outlet_Size[mart_data.Outlet_Size.isnull()] = 'Medium'
mart_data.apply(lambda x : sum(x.isnull()))
plt.scatter(x = 'Item_Weight',y = 'Item_Outlet_Sales', data = mart_data)
plt.show();
plt.hist(x = mart_data['Item_Weight'].dropna(),data = mart_data,bins = 20)
plt.show();
mart_data['Item_Weight'].describe()
import seaborn as sns
sns.boxplot(x = 'Item_Weight', data = mart_data)
Item_weight_group = mart_data.groupby('Item_Identifier')[['Item_Identifier','Item_Weight']].mean()
Item_weight_group.head()
mart_data['Item_Weight'] = mart_data.groupby('Item_Identifier')['Item_Weight'].transform('mean')
mart_data.isnull().sum()
mart_data.loc[mart_data.Item_Type == 'Non-Consummable','Item_Fat_Content'] = 'Non-Edible'
mart_data['Item_Fat_Content'].value_counts()
print('Number of rows with Zero Visibility: ',mart_data[mart_data['Item_Visibility'] == 0]['Item_Identifier'].count())
#Determine average visibility of a product
visibility_avg = mart_data.pivot_table(values='Item_Visibility', index='Item_Identifier')
visibility_avg.head()
def visibility(x):
    if (visibility_avg.index == x).any():
        return(visibility_avg.loc[x,'Item_Visibility'])
    else:
        return(0)
mart_data.loc[mart_data.Item_Visibility == 0,'Item_Visibility'] = mart_data.loc[mart_data.Item_Visibility == 0,
                                                                                'Item_Identifier'].apply(lambda x: visibility(x))

mart_data[mart_data['Item_Visibility'] == 0].count()
mart_data.info()
mart_data['Item_Fat_Content'] = mart_data['Item_Fat_Content'].astype('category')
mart_data['Item_Type'] = mart_data['Item_Type'].astype('category')
mart_data['Outlet_Location_Type'] = mart_data['Outlet_Location_Type'].astype('category')
mart_data['Outlet_Size'] = mart_data['Outlet_Size'].astype('category')
mart_data['Outlet_Type'] = mart_data['Outlet_Type'].astype('category')
sns.boxplot('Item_Outlet_Sales',data = mart_data)
sns.distplot(mart_train['Item_Outlet_Sales'])
plt.show();
# Distribution of Item_MRP andd their sales
plt.scatter(x = 'Item_MRP', y = 'Item_Outlet_Sales',data = mart_data)
plt.title('Distribution of Sales vs Item_MRP',fontsize = 18)
plt.xlabel('Item_MRP',fontsize = 14)
plt.ylabel('Item_sales',fontsize = 15)
plt.show();
plt.figure(figsize=(10,6))
plt.title('Pearson Correlation of Features', y = 1, size = 20)
sns.heatmap(mart_data.loc[:,mart_data.dtypes == float].corr(),linewidths=0.1,vmax=1.0,square= True,
           cmap = plt.cm.RdBu_r, linecolor = 'white', annot = True)
plt.xticks(rotation = 0)
plt.show();
#Looking at the distribution of sales across categories
plt.figure(figsize=(14,8))
plt.subplot(221)
sns.boxplot(x = 'Item_Type' , y = 'Item_Outlet_Sales', data = mart_data)
plt.subplot(222)
sns.boxplot(x = 'Outlet_Type' , y = 'Item_Outlet_Sales', data = mart_data)
plt.subplot(223)
sns.boxplot(x = 'Outlet_Size' , y = 'Item_Outlet_Sales', data = mart_data)
plt.subplot(224)
sns.boxplot(x = 'Outlet_Location_Type' , y = 'Item_Outlet_Sales', data = mart_data)
plt.show();
