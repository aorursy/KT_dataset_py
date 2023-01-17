# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

item_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

train.head()
test.head()
test_df = test

train_df = train
test_df['date_block_num']=0

test_df['item_cnt_day']=0.285  



train_df['date_block_num'] = 33 - train_df['date_block_num']



train_df['date'] = pd.to_datetime(train_df.date,format="%d.%m.%Y")

itemrange= pd.DataFrame( (train_df.groupby('item_id').date.max()-train_df.groupby('item_id').date.min() ).dt.days ) 





train_df=train_df.append(test_df.drop('ID',axis=1),sort=False )



train_df=train_df.merge(items,how='left',left_on='item_id',right_on='item_id')

train_df=train_df.merge(itemrange,how='left',left_index=True,right_index=True)



def piv_clust(data,veld,kolom,waarde,komponent):

    from sklearn.decomposition import TruncatedSVD,FastICA

    df = data.pivot_table(index=veld, columns=kolom, values=waarde, fill_value=0, aggfunc=np.sum)         

    svd = TruncatedSVD(n_components=komponent, n_iter=7, random_state=42)                                 

    ica = FastICA(n_components=komponent)

    df_norm =( (df - df.mean()) / (df.max() - df.min()) ).fillna(0)                                       

    return pd.DataFrame( np.concatenate([svd.fit_transform(df_norm)*svd.singular_values_, ica.fit_transform(df_norm)],axis=1) , index=df.index),svd.explained_variance_ratio_  #U*S



train_p = train_df.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', fill_value=0, aggfunc=np.sum)





train_i,sing_i=piv_clust(train_df,'item_id',['shop_id','date_block_num'],'item_price',10)

train_s,sing_s=piv_clust(train_df,'shop_id',['item_category_id','date_block_num'],'item_price',10)

train_si,sing_si=piv_clust(train_df[train_df['date_block_num']>10],'item_id',['date_y'],'item_cnt_day',10)

train_si
import matplotlib.pyplot as plt

import seaborn as sns



d = {}

for i in range(len(item_cat)): 

    d[i] = item_cat['item_category_name'][i]

            

items['cat_name'] = ''

for j in range(len(items)):

    items['cat_name'][j] = d[int(items['item_category_id'][j])] 
items.head()
top_10 = items.groupby(['cat_name']).count().sort_values('item_id', ascending=False).iloc[0:9].reset_index()

plt.figure(figsize=(20,10))

plt.bar(top_10.cat_name, top_10.item_id)

plt.title("Top 10 category")

plt.ylabel('count')

plt.xticks(rotation=15)

plt.show()
d = {}

for i in range(len(items)): 

    d[i] = items['item_name'][i]

for j in range(len(items)):

    items['cat_name'][j] = d[int(items['item_category_id'][j])] 

top_10 = train.groupby(['item_id']).count().sort_values('date', ascending=False).iloc[0:10].reset_index()



top_10['item_name'] = ''

for j in range(len(top_10)):

    top_10['item_name'][j] = d[int(items['item_id'][j])] 





plt.figure(figsize=(20,10))

plt.bar(top_10.item_name, top_10.date)

plt.title("Top 10 items by sales")

plt.ylabel('count')

plt.xticks(rotation=90)

plt.show()
def submfi(tr_te,test,pre,file):                                          

    tr_te['pre']=pre

    s_df=tr_te[['pre']].clip(0)           

    s_df=s_df.merge(test,how='right',left_index=True,right_on=['shop_id','item_id'])

    s_df=s_df[['pre','ID']]               

    s_df.columns=['item_cnt_month','ID']  

    s_df.to_csv(file, index=False)        

    return
from sklearn.linear_model import LinearRegression

regr = LinearRegression()



for xi in range(26,27):

    

    x = train_p.iloc[:,1:].clip(0,xi)

    y = train_p.iloc[:,0].clip(0,20)

    regr.fit(x, y)

    

    pred = regr.predict(x)

    

    print('Intercept: \n', regr.intercept_)

    print('Coefficients: \n', regr.coef_)

    

    print(xi,'MSRE: %.4f'%np.sqrt(((y-pred)*(y-pred)).mean()))

    

p=regr.predict(train_p.iloc[:,0:-1].clip(0,xi).fillna(0))

submfi(train_p,test_df,p,'subm0.csv')