# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import roc_curve , auc , mean_squared_error
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#load Data
df=pd.read_csv('../input/sales_train.csv')
#Date Format
df.date=df.date.apply(lambda x:datetime.strptime(x, '%d.%m.%Y'))

len(df),df.head()
#Load Items
items=pd.read_csv('../input/items.csv')
#Merge with Data
df_final=df.merge(items,on='item_id',how='left')
#Create Total Revenue Feature
df_final['sales']=df_final.item_cnt_day * df_final.item_price
df_final.shape
new_columns=['date','date_block_num','item_id','item_name','item_category_id','shop_id','item_price','item_cnt_day','sales']
#Arrange Columns
df_final=df_final.reindex(new_columns,axis=1)
df_final.head()
#Create Month as a new feature
df_final['month']=df_final.date.apply(lambda x: x.strftime("%m"))

#Create Year as a new feature
df_final['year']=df_final.date.apply(lambda x: x.strftime("%Y"))

#Item Price Outliers
plt.boxplot(df_final.item_price)
df_final[df_final.item_price>150000]
#One Row to be removed
df_final=df_final[df_final.item_price<150000]
#Check for other outliers
plt.boxplot(df_final.item_price)

df_final[df_final.item_id.isin(df_final.item_id[df_final.item_price>40000])].groupby('item_id').agg({'item_id': 'count' , 'item_price': 'mean'})
df_final=df_final[df_final.item_price<40000]
plt.boxplot(df_final.item_price)
plt.ylabel('Item Price')
plt.grid()
#Check negative item price
df_final[df_final.item_price<0]
#Remove negative item price records
df_final=df_final[df_final.item_price>=0]
#Check item_cnt_day outliers 
plt.boxplot(df_final.item_cnt_day)
plt.ylabel('Item Cnt Day')
plt.grid()
df_final[df_final.item_id.isin(df_final.item_id[df_final.item_cnt_day>800])].groupby('item_id').agg({'item_id': 'count' , 'item_cnt_day': 'mean'})
#Clearly Outliers and should be removed
df_final=df_final[df_final.item_cnt_day<=900]
plt.boxplot(df_final.item_cnt_day)
plt.ylabel('Item Cnt Day')
plt.grid()

df_items=set(df_final.item_id.values)

test_df=pd.read_csv('../input/test.csv')
test_item=set(test_df.item_id.values)
print (str(len(test_item)))
diff_item=test_item-df_items
len(diff_item)
diff_item=list(diff_item)
#Get List of Item Categories for items not in training dataset
diff_item_category=items.item_category_id[items.item_id.isin(diff_item)]
diff_item_category=pd.DataFrame(diff_item_category)
#For below list , prediction will be only through Item_category Cnt, and Item_category_price
diff_item_category.groupby('item_category_id')['item_category_id'].count()
df_group=df_final.groupby([col for col in df_final if col not in ['date','item_price','item_cnt_day','sales']],as_index=False)['sales','item_cnt_day'].agg({"sales":"sum","item_cnt_day":"sum"})
df_group.rename(columns={'item_cnt_day':'item_cnt_month', 'sales':'sales_month'},inplace=True)
df_group.head() 
df_group[df_group.item_cnt_month==0]
df_group['mean_item_shop_month_price']= df_group[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
df_group.head()
df_group[df_group.item_cnt_month==0]
#Get Item_category_per shop list
item_category_shop_df=df_group.groupby(['date_block_num','item_category_id','shop_id'],as_index=False).agg({'item_cnt_month':'sum','sales_month': 'sum'})
item_category_shop_df1=df_group.groupby(['date_block_num','item_category_id','shop_id'],as_index=False).agg({'item_cnt_month':'count'})
item_category_shop_df['mean_category_shop_month_cnt']=item_category_shop_df[['item_cnt_month']]/item_category_shop_df1[['item_cnt_month']]     
item_category_shop_df['mean_ctegory_shop_month_price']= item_category_shop_df[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
item_category_shop_df.drop(columns=['sales_month','item_cnt_month'],axis=1,inplace=True)
item_category_shop_df.head()
#Will Do the same but for generic item_category without specifiying date blook num
item_category_shop_all=df_group.groupby(['item_category_id','shop_id'],as_index=False).agg({'item_cnt_month':'sum','sales_month': 'sum'})
item_category_shop_all1=df_group.groupby(['item_category_id','shop_id'],as_index=False).agg({'item_cnt_month':'count'})
item_category_shop_all['mean_category_shop_cnt']=item_category_shop_all[['item_cnt_month']]/item_category_shop_all1[['item_cnt_month']]
item_category_shop_all['mean_ctegory_shop_price']= item_category_shop_all[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
item_category_shop_all.drop(columns=['sales_month','item_cnt_month'],axis=1,inplace=True)
item_category_shop_all.head(20)
#Will Do the same but for generic item_category without specifiying date blook num
item_category_all=df_group.groupby(['item_category_id'],as_index=False).agg({'item_cnt_month':'sum','sales_month': 'sum'})
item_category_all1=df_group.groupby(['item_category_id'],as_index=False).agg({'item_cnt_month':'count'})
item_category_all['mean_category_cnt']=item_category_all[['item_cnt_month']]/item_category_all1[['item_cnt_month']]
item_category_all['mean_ctegory_price']= item_category_all[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
item_category_all.drop(columns=['sales_month','item_cnt_month'],axis=1,inplace=True)
item_category_all.head()
#Will Do the same but for generic item_category without specifiying date blook num
item_shop_all=df_group.groupby(['item_id','shop_id'],as_index=False).agg({'item_cnt_month':'sum','sales_month': 'sum'})
item_shop_all1=df_group.groupby(['item_id','shop_id'],as_index=False).agg({'item_cnt_month':'count'})
item_shop_all['mean_item_shop_cnt']=item_shop_all[['item_cnt_month']]/item_shop_all1[['item_cnt_month']]
item_shop_all['mean_item_shop_price']= item_shop_all[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
item_shop_all.drop(columns=['sales_month','item_cnt_month'],axis=1,inplace=True)
item_shop_all.head()
#Will Do the same but for generic item_category without specifiying date blook num
item_all=df_group.groupby(['item_id'],as_index=False).agg({'item_cnt_month':'sum','sales_month': 'sum'})
item_all1=df_group.groupby(['item_id'],as_index=False).agg({'item_cnt_month':'count'})
item_all['mean_item_cnt']=item_all[['item_cnt_month']]/item_all1[['item_cnt_month']]
item_all['mean_item_price_all']= item_all[['sales_month','item_cnt_month']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0,axis=1)
item_all.drop(columns=['sales_month','item_cnt_month'],axis=1,inplace=True)
item_all.head()
df_group=df_group.merge(item_category_shop_df,how='left',on=['date_block_num','item_category_id','shop_id']).fillna(0)
df_group=df_group.merge(item_category_shop_all,how='left',on=['item_category_id','shop_id']).fillna(0)
df_group=df_group.merge(item_category_all,how='left',on=['item_category_id']).fillna(0)
df_group=df_group.merge(item_shop_all,how='left',on=['item_id','shop_id']).fillna(0)
df_group=df_group.merge(item_all,how='left',on=['item_id']).fillna(0)
df_group.head()
sns.distplot(np.log1p(df_group.item_cnt_month[df_group.item_cnt_month>0]))
cat_var=['item_category_id', 'shop_id',
       'month', 'year']

fig , ax = plt.subplots(2,2,figsize=(20,10))
for i , x in zip(ax.flat,cat_var):
    sns.boxplot(x=df_group[x],y=np.log1p(df_group.item_cnt_month),ax=i)
    i.set(ylabel='log item cnt month')

cont_var=['mean_item_shop_month_price', 'mean_category_shop_month_cnt',
       'mean_ctegory_shop_month_price', 'mean_category_shop_cnt',
       'mean_ctegory_shop_price', 'mean_category_cnt', 'mean_ctegory_price',
       'mean_item_shop_cnt', 'mean_item_shop_price', 'mean_item_cnt',
       'mean_item_price_all']
fig , ax = plt.subplots(3,3,figsize=(20,10))
for i , x in zip(ax.flat,cont_var):
    sns.scatterplot(x=np.log1p(df_group[x]),y=np.log1p(df_group.item_cnt_month),ax=i)
    i.set(xlabel='log ' + str(df_group[x].name),ylabel='log item_cnt_month')
    
test_df=pd.read_csv('../input/test.csv')
test=test_df
#test.drop(columns='Unnamed: 0',axis=1,inplace=True)
test['date_block_num']=34
test['month']=11
test['year']=2015
test.head()
test=test.merge(items,on='item_id',how='left')
test.head()
test=test.merge(item_category_shop_df,how='left',on=['date_block_num','item_category_id','shop_id']).fillna(0)
test=test.merge(item_category_shop_all,how='left',on=['item_category_id','shop_id']).fillna(0)
test=test.merge(item_category_all,how='left',on=['item_category_id']).fillna(0)
test=test.merge(item_shop_all,how='left',on=['item_id','shop_id']).fillna(0)
test=test.merge(item_all,how='left',on=['item_id']).fillna(0)
test.head()
df_group.drop(columns=['sales_month'],axis=1,inplace=True)
#Concatenate train , test together for more feature engineering
model=pd.concat([df_group,test.iloc[:,1:]],ignore_index=True,sort=False)
model.drop(columns='item_name',axis=1,inplace=True)
model.tail()
#Function to get previous months attributes
def lag(df,lags,col):
    x=df[['date_block_num','item_id','shop_id',col]]
    for i in lags:
        y=x.copy()
        y['date_block_num']=y['date_block_num']+i
        y.rename(columns={col : col+'_lag_'+str(i)},inplace=True)
        print(y.columns)
        
        df=df.merge(y,how='left',on=['date_block_num','item_id','shop_id'])
    return df

def impute_na(df,col):
    for i in col:
        print('Numer to Impute ',i,':',str(df[i].isna().sum()))
        df[i][df[i].isna()]=0    
    return df

def quarter_mean(df,col):
    return(df[col].apply(lambda x: np.mean(x),axis=1))

        
#Applying function to add more features for lag_item_cnt_month
model=lag(model,[1,2,3],'item_cnt_month')
#Applying function to add more features for lag_mean_item_price
model=lag(model,[1,2,3],'mean_category_shop_month_cnt')
model=lag(model,[1],'mean_item_shop_month_price')
model=impute_na(model,['item_cnt_month_lag_1',
       'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'mean_category_shop_month_cnt_lag_1',
       'mean_category_shop_month_cnt_lag_2','mean_category_shop_month_cnt_lag_3',
       'mean_item_shop_month_price_lag_1'])
model.tail()

model['quarnter_cnt_month_lag_1']=quarter_mean(model,['item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3'])
model['quarnter_mean_category_shop_month_cnt_lag_1']=quarter_mean(model,['mean_category_shop_month_cnt_lag_1','mean_category_shop_month_cnt_lag_2','mean_category_shop_month_cnt_lag_3'])
model=model.drop(columns=['mean_item_shop_month_price','mean_category_shop_month_cnt','mean_ctegory_shop_month_price',
                   'item_cnt_month_lag_2','item_cnt_month_lag_3','mean_category_shop_month_cnt_lag_1','mean_category_shop_month_cnt_lag_2',
                   'mean_category_shop_month_cnt_lag_3','mean_category_shop_month_cnt_lag_1','mean_category_shop_month_cnt_lag_2','mean_category_shop_month_cnt_lag_3'
                   ],axis=1)
model.tail()
def log_cont_var(df,col):
    for column in col:
        df[column+'_log']=np.log1p(df[column])
    df=df.drop(columns=col,axis=1)
    return df

def clip_negative(df):
    col=df.columns[df.dtypes=='float64']
    for column in col:
        df[column][df[column]<0]=0
    return df

model= clip_negative(model)
cont_var=['item_cnt_month', 'mean_category_shop_cnt',
       'mean_ctegory_shop_price', 'mean_category_cnt', 'mean_ctegory_price',
       'mean_item_shop_cnt', 'mean_item_shop_price', 'mean_item_cnt',
       'mean_item_price_all', 'item_cnt_month_lag_1','mean_item_shop_month_price_lag_1',
       'quarnter_cnt_month_lag_1',
       'quarnter_mean_category_shop_month_cnt_lag_1']

model=log_cont_var(model,cont_var)
model.tail()
#Save Dataset for Running models
model.tail()

