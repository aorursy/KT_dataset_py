import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Import the dataset
train = pd.read_csv('../input/dsn-ai-oau-july-challenge/train.csv')
test = pd.read_csv('../input/dsn-ai-oau-july-challenge/test.csv')
sub = pd.read_csv('../input/dsn-ai-oau-july-challenge/sample_submission.csv')
train.head()
train.describe()
train.isna().sum()
train.Product_Weight.fillna(12.6, inplace=True)
train.head()
#train.Supermarket_Opening_Year.value_counts()
def bar_plot(df,x,y):
    #plt.figure(figsize=(5,3))
    df.groupby(x).mean()[y].sort_values(ascending=False).plot(kind='barh')
    #plt.xlabel(y, fontsize=6)
   # plt.ylabel(x, fontsize=6)
    #plt.tick_params(labelsize=30)
    plt.title('Average ' + y + ' based on ' + x  )
    plt.tight_layout()
bar_plot(train,'Product_Fat_Content','Product_Weight' )
bar_plot(train,'Product_Type','Product_Weight' )
bar_plot(train,'Product_Type','Product_Price' )
train.groupby('Product_Type')['Product_Type'].count().sort_values(ascending=False).plot(kind='barh')
plt.title('Count of Product Type');
train.groupby('Supermarket_Type')['Supermarket_Type'].count().sort_values(ascending=False).plot(kind='barh')
plt.title('Count of Supermarket Type');
train.groupby('Supermarket_Location_Type')['Supermarket_Location_Type'].count().sort_values(ascending=False).plot(kind='barh')
plt.title('Count of Supermarket_Location_Type');
def dist_plot(col):
    sns.distplot(train[col], color='r')
    plt.title('Distribution plot of ' + col)
    plt.show()
    
    
dist_plot('Product_Weight')
dist_plot('Product_Shelf_Visibility')
#train.Product_Weight.value_counts()
dist_plot('Product_Price')
def count(col):
    sns.countplot(train[col])
    plt.title('Count of '+ col)
    plt.show()
count("Product_Fat_Content")
#train.info()
#count('Product_Type')
#plt.tight_layout()
#plt.plot(train['Product_Type'], kind='bar')
count('Supermarket _Size')
count('Supermarket_Location_Type')
#count('Supermarket_Type')
#plt.tight_layout()
#train.groupby('Supermarket_Type')['Supermarket_Type'].count().sort_values(ascending=False).plot(kind='barh')
count('Supermarket_Opening_Year')
#train.groupby('Supermarket_Opening_Year')['Supermarket_Opening_Year'].mean().plot(kind='barh')
#train.info()
def scatter_plot(a,b, data=train):
    sns.scatterplot(train[a],train[b])
    plt.title(f'Scatter plot of {a} vs {b} ')
scatter_plot('Product_Weight','Product_Price')
scatter_plot('Product_Weight','Product_Shelf_Visibility')
scatter_plot('Product_Price','Product_Shelf_Visibility')
def bar_plot(x,y,z, data=train):
    sns.barplot(train[x], train[y], hue=train[z])
    plt.title(x + ' vs ' + y + ' based on ' + z)

plt.figure(figsize=(8,4))
bar_plot('Supermarket _Size','Product_Weight','Product_Fat_Content')
plt.tight_layout()
plt.figure(figsize=(8,4))
bar_plot('Supermarket_Location_Type','Product_Weight','Product_Fat_Content')
plt.tight_layout()
plt.figure(figsize=(10,5))
bar_plot('Supermarket_Location_Type','Product_Price','Product_Fat_Content')
plt.tight_layout()
train.groupby('Supermarket_Type')['Product_Supermarket_Sales'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Supermarket sales based on supermarket type');
train.groupby(['Supermarket_Type','Product_Type'])['Product_Supermarket_Sales'].mean().loc['Supermarket Type3'].sort_values(ascending=False).plot(kind='barh')
plt.title('Product Type with most sales in Supermarket 3')
train.groupby(['Supermarket_Type','Product_Type'])['Product_Supermarket_Sales'].mean().loc['Supermarket Type1'].sort_values(ascending=False).plot(kind='barh')
plt.title('Product Type with most sales in Supermarket 1');
train.groupby(['Supermarket_Type','Product_Type'])['Product_Supermarket_Sales'].mean().loc['Supermarket Type1'].sort_values(ascending=False).plot(kind='barh')
plt.title('Product Type with most sales in Supermarket 1')
train.groupby(['Supermarket_Type','Product_Type'])['Product_Supermarket_Sales'].mean().loc['Grocery Store'].sort_values(ascending=False).plot(kind='barh')
plt.title('Product Type with most sales in Grocery Store')
train.groupby('Supermarket_Location_Type')['Product_Supermarket_Sales'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Supermarket sales based on supermarket location type');
train.groupby('Supermarket _Size')['Product_Supermarket_Sales'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Supermarket sales based on supermarket size');
train.groupby('Product_Type')['Product_Supermarket_Sales'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Supermarket sales based on product type');
train.groupby('Product_Type')['Product_Shelf_Visibility'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Shelf Visibility sales based on Product Type');
train.groupby('Product_Type')['Product_Price'].mean().sort_values(ascending=False).plot(kind='barh')
plt.title('Product Price based on Product Type');

