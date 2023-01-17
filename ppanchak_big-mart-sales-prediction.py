import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
train='../input/Train.csv'
train_set1=pd.read_csv(train)
test='../input/Test.csv'
test_set=pd.read_csv(test)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
train='../input/Train.csv'
train_set1=pd.read_csv(train)
test='../input/Test.csv'
test_set=pd.read_csv(test)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
train_set1.shape
test_set.shape
train_set=pd.concat([train_set1,test_set])
train_set.head(5)
train_set.describe()
train_set.info()
train_set.isnull().sum()
for cols in train_set.columns:
    if (train_set[cols].dtype=='object'):
        print(train_set[cols].value_counts())
#Checking for the availability of items in outlets
item_count=train_set['Item_Identifier'].value_counts().sort_values()
plt.hist(item_count,bins=np.arange(0,11,1))
train_set['Item_Identifier'].describe()
plt.show()
item_count_df=pd.DataFrame(item_count)
item_cnt=(item_count_df['Item_Identifier'].value_counts()/sum(item_count_df['Item_Identifier'])*100)
plt.xlabel('Number of items in outstores')
item_count_df['Item_Identifier'].value_counts().plot(kind='bar',color='orange')

#The below graph shows the availabilty and distribution of items in number of stores.
#As we can see that there are 13 items that are available only in 7 store, likewise 305 items in 8,504 items in 10, 737 in 9 stores
# Item Weight

# We can see "Item Weight" has missing values. From the distribution in data we can see that there are hikes 
# in between and even falls
train_set['Item_Weight'].describe()
train_wtou_null=train_set['Item_Weight'].dropna()
plt.hist(train_wtou_null,bins=30)
plt.show()
# Item_Fat_Content

# We can see the inconsistency in data, as there are only 2 unique categories as "Low Fat" and "Regular"
# We will need to convert "LF" and "low fat" into "Low Fat" and "reg" into "Regular"
# From the bar chart, we can see that about 65% of the items have low fat.
# That is people perfer purchasing low fat Item

train_set['Item_Fat_Content'].value_counts()
train_set.loc[train_set['Item_Fat_Content']=='LF','Item_Fat_Content']='Low Fat'
train_set.loc[train_set['Item_Fat_Content']=='low fat','Item_Fat_Content']='Low Fat'
train_set.loc[train_set['Item_Fat_Content']=='reg','Item_Fat_Content']='Regular'
content_cnt=pd.DataFrame(train_set.groupby(['Item_Identifier','Item_Fat_Content'])['Item_Fat_Content'].count())
content_cnt=content_cnt.rename(columns={'Item_Fat_Content':'Cnt'}).reset_index()
content_cnt['Item_Fat_Content'].value_counts()/sum(content_cnt['Item_Fat_Content'].value_counts())
(content_cnt['Item_Fat_Content'].value_counts()/sum(content_cnt['Item_Fat_Content'].value_counts())).plot(kind='bar')
#Item Visibility
#we can see that most of the items have visibility less that 5%,There are few outliers and see that the distribution is rightskewed
#calculating the 95th and 99th percentiles,

b,c,d= np.percentile(train_set['Item_Visibility'],[90,95,99])
print(b,c,d)
train_set['Item_Visibility'].describe()
train_set['Item_Visibility'].hist(bins=20)
# If we from the below chart we can see that average visibility Small Outlet is more as compared to other outlet size 
train_set.groupby('Outlet_Size')['Item_Visibility'].mean()
sns.scatterplot(data=train_set,y='Item_Type',x='Item_Visibility',hue='Outlet_Size')
# fromthe below histogram we can that it a multimodal distribution 
train_set['Item_MRP'].describe()
train_set['Item_MRP'].hist(bins=30)

#Most of the item sold belongs to category "Fruits and Vegetables" and "Snack Foods".
# Seafood have low demand
train_set['Item_Type'].value_counts().plot(kind='bar',color='orange')
# By considering the combination of item and item type, will give us the unique count of Item type 
type_cnt=pd.DataFrame(train_set.groupby(['Item_Identifier','Item_Type'])['Item_Type'].count())
type_cnt=type_cnt.rename(columns={'Item_Type':'Cnt'}).reset_index()
type_cnt['Item_Type'].value_counts()/sum(type_cnt['Item_Type'].value_counts())*100
(type_cnt['Item_Type'].value_counts()/sum(type_cnt['Item_Type'].value_counts())*100).plot(kind='bar',color='orange')
#From the below analysis we can say that outlet_identifier 'OUT010' and 'OUT019' are Grocery Stores.Hence the
# chances of getting the items belonging to some specific catgories decreases 
train_set['Outlet_Identifier'].value_counts().plot(kind='bar',color='orange')
#train_set.sort_values(by='Item_Identifier')
train_set.groupby(['Outlet_Identifier','Outlet_Type'])['Outlet_Identifier'].unique()
# Outlet_Establishment_Year.
# Converting the Outlet_Establishment_Year in Date Format
train_set['Outlet_Establishment_Year']=train_set['Outlet_Establishment_Year'].astype(str).apply(lambda x: pd.to_datetime(x, format='%Y%'))
train_set['Year']=train_set['Outlet_Establishment_Year'].dt.year
train_set['Year'].value_counts()
train_set['Outlet_Size'].value_counts()
train_set['Outlet_Location_Type'].value_counts()
train_set['Outlet_Type'].value_counts()
train_set.groupby(['Outlet_Location_Type','Outlet_Type','Outlet_Size'])['Outlet_Identifier'].count()
# We can see that there are outliers present in data
train_set['Item_Outlet_Sales'].hist()
train_set['Item_Outlet_Sales'].describe()
outlet_size_mode= train_set.pivot_table(values='Outlet_Size', columns = 'Outlet_Type',aggfunc=lambda x: x.mode())
outlet_size_mode
def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
    
train_set['Outlet_Size'] = train_set[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
item_wt=train_set.pivot_table(values='Item_Weight',columns='Item_Identifier',aggfunc='mean')

def itm_wt(cols):
    itm=cols[0]
    wt=cols[1]
    if(pd.isnull(wt)):
        return item_wt.loc['Item_Weight'][item_wt.columns== itm][0]
    else:
        return wt
    
train_set['Item_Weight']=train_set[['Item_Identifier','Item_Weight']].apply(itm_wt,axis=1)
#We also found that the visibility of the item cannot be zero hence, we can subsitute the mean of 'Item Visibility' where 
# Item visibility is zero
# We can see the hike in the chart as we imputed the mean.

train_set.loc[train_set['Item_Visibility']==0.000000,'Item_Visibility']=np.mean(train_set['Item_Visibility'])
sns.distplot(train_set['Item_Visibility'],bins=50)
plt.show()

train_set.isnull().sum()
# We can see that the net sales of the regular fat item is slightly high.
pd.pivot_table(data=train_set,columns='Item_Fat_Content',values='Item_Outlet_Sales',aggfunc='mean').plot(kind='bar')
pd.pivot_table(data=train_set,columns='Item_Fat_Content',values='Item_MRP',aggfunc='mean')
train_set.groupby(['Item_Fat_Content'])['Item_Identifier'].count()
stats.spearmanr(train_set['Item_Fat_Content'],train_set['Item_MRP'])

sales_loc=pd.pivot_table(data=train_set,index=['Item_Fat_Content','Outlet_Location_Type'],values='Item_Outlet_Sales',aggfunc='mean').reset_index()
sales_loc
plt.figure(figsize=(5,5))
sns.barplot(data=sales_loc,x='Item_Fat_Content',hue='Outlet_Location_Type',y='Item_Outlet_Sales')
plt.figure(figsize=(8,8))
plt.subplot(2,1,2)
train_set['Item_Type'].value_counts().plot(kind='bar',color='orange')
pd.pivot_table(data=train_set,index='Item_Type',values=['Item_Outlet_Sales','Item_MRP'],aggfunc='mean')
plt.subplot(2,1,1)
sns.barplot(data=train_set,y='Item_Type',x='Item_Outlet_Sales',color='orange')
#if we see the average sales,tier 2,3 have more average sales as compared to other cities. It may be due to tier 2,tier 3 are moving to
#towards adopting new products, may be the products have more discounts and sales may increase
loc_sales=pd.pivot_table(data=train_set,index=['Outlet_Location_Type'],values='Item_Outlet_Sales',aggfunc='mean').reset_index()
loc_sales
plt.figure(figsize=(8,8))
ax = sns.boxplot(x="Outlet_Location_Type", y="Item_Outlet_Sales", data=train_set)
#sns.boxplot(data=loc_sales,x='Item_Outlet_Sales',hue='Outlet_Location_Type')
train_set.isnull().sum()
train_set.groupby(['Outlet_Identifier','Outlet_Type','Outlet_Size'])['Item_Outlet_Sales'].mean().plot(kind='bar',color='blue')
#outlet size doesnt make more difference in the sales, if we compare the sales outlet type wise, then Supermarket Type1 with 
#size : [Small,Medium,high] shows almost same average sales. 
#As the OUT010 and OUT019 are small grocery store,the number of items available are less as compared to othe stores 
#therefore tha average sales of this outlets is less
train_set.groupby(['Outlet_Identifier','Outlet_Size','Outlet_Location_Type'])['Item_Identifier',].count()
train_set.groupby(['Outlet_Identifier'])['Item_Identifier'].count().plot(kind='bar',color='orange')
# if we see that average sales for the item, it's pretty high in Tier 3 city, this can be due to people from this cities
#moving to desire and try on new products that are available in supermarket of Tier 1 cities
train_set.groupby(['Outlet_Size','Outlet_Location_Type'])['Item_Outlet_Sales'].mean().plot(kind='bar',color='blue')
ItemType_sales=pd.pivot_table(data=train_set,index=['Item_Type','Outlet_Identifier'],values='Item_Outlet_Sales',aggfunc='mean').reset_index()
plt.figure(figsize=(15,15))
g=sns.FacetGrid(data=train_set,row='Outlet_Identifier',aspect=2)
g=g.map(plt.bar,'Item_Type','Item_Outlet_Sales')
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
#Adding an Item Category Variable by observing the Item Identifier,
def create_Category(data):
    if str(data['Item_Identifier']).startswith('NC'):
        return 'Non Consumable'
    elif str(data['Item_Identifier']).startswith('FD'):
        return 'Food'
    else:
        return 'Drinks'
    #train_set['check1']= train_set['Item_Identifier'].apply(lambda x: x[0:2])

train_set['Item_Category']= train_set.apply(create_Category,axis=1)

train_set['Item_Category'].value_counts()
cat_sales=pd.pivot_table(data=train_set,index=['Item_Category','Outlet_Identifier','Outlet_Type'],values='Item_Outlet_Sales',aggfunc='mean').reset_index()
g=sns.FacetGrid(data=cat_sales,col='Item_Category',hue='Outlet_Type',legend_out=True)
g=g.map(plt.bar,'Outlet_Identifier','Item_Outlet_Sales')

for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
# After  dividing the items in to item category, we can find that category "Non consumable" will have no fats hence 
# adding a level to Item_Fat_Content as "No Fats" 

train_set.loc[train_set['Item_Category']=='Non Consumable','Item_Fat_Content']='No Fat'
train_set['Item_Fat_Content'].value_counts()
train_set['Visibility_log']=np.log10(train_set['Item_Visibility'])
plt.hist(train_set['Visibility_log'])
train_set=pd.get_dummies(train_set,columns=['Item_Fat_Content'])
#train_set=pd.concat([train_set,train_fat_content])
train_set=pd.get_dummies(train_set,columns=['Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Category'])
list1=['Item_MRP','Item_Visibility', 'Item_Weight',
       'Year', 'Visibility_log', 'Item_Fat_Content_Low Fat',
       'Item_Fat_Content_No Fat', 'Item_Fat_Content_Regular',
       'Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Identifier_OUT010', 'Outlet_Identifier_OUT013',
       'Outlet_Identifier_OUT017', 'Outlet_Identifier_OUT018',
       'Outlet_Identifier_OUT019', 'Outlet_Identifier_OUT027',
       'Outlet_Identifier_OUT035', 'Outlet_Identifier_OUT045',
       'Outlet_Identifier_OUT046', 'Outlet_Identifier_OUT049',
       'Outlet_Location_Type_Tier 1', 'Outlet_Location_Type_Tier 2',
       'Outlet_Location_Type_Tier 3', 'Outlet_Size_High', 'Outlet_Size_Medium',
       'Outlet_Size_Small', 'Outlet_Type_Grocery Store',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3', 'Item_Category_Drinks',
       'Item_Category_Food', 'Item_Category_Non Consumable','Item_Outlet_Sales']
train_set=train_set[list1]
#dividing the train_set into training and test set
train=train_set.loc[train_set['Item_Outlet_Sales'] > 0]
test=train_set.loc[train_set['Item_Outlet_Sales'].isnull()]
train.shape
test.shape
train.head(2)
test.head(2)
del test['Item_Outlet_Sales']
train.shape
test.shape
#preparing the Validation data 
X_train=train.iloc[:, :-1]
Y_train=train.iloc[:,-1]
test.shape
from sklearn.linear_model import LinearRegression

regresor= LinearRegression()
regresor.fit(X_train,Y_train)
y_pred=regresor.predict(test)
y_pred1=pd.DataFrame(y_pred)
y_pred1.to_csv('pred', sep='\t', encoding='utf-8')