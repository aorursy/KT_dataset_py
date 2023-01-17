import numpy as np

import pandas as pd

from pandas import Series, DataFrame
df = pd.read_csv('../input/ad_feature.csv', sep=',', header=0)
df.shape
df.columns
df.head(10)
df[:100].to_csv('output.csv', index=False)    # output to csv without inde
df[:100].to_html('output.html', index=False)    # output to website table
writer = pd.ExcelWriter('output.xlsx')    # output to excel

df[:100].to_excel(writer, 'Sheet1', index=False)

df[100:200].to_excel(writer, 'Sheet2', index=False)

writer.save()
df.head(5)    # first 5 rows
df.tail(5)    # last 5 rows
df['price'].describe()  
df['cate_id'].describe() 
# Iteration

for row in df[:10].itertuples():

    print(row.cate_id, row.price)
# Sort

df[:10].sort_values(by='price', ascending=False) 
df[:10].sort_values(by=['price', 'brand'], ascending=False) 
df[(df['price'] > 1000) & (df['price'] <= 1005)]
df = pd.read_csv('../input/ad_feature.csv', sep=',', header=0, nrows=1000)

cate_group = df.groupby('cate_id')    

type(cate_group)
cate_group.size()
cate_group['price', 'brand'].mean().head()    # Average price and brand by cate_id
cate_group['price', 'brand'].mean().reset_index(drop=False).head()    
df = pd.read_csv('../input/ad_feature.csv', sep=',', header=0)

df1 = df[['adgroup_id', 'price']]

df2 = df[['adgroup_id', 'cate_id', 'brand']]
df1.head()
df2.head()
df3 = pd.merge(df1, df2, on='adgroup_id', how='inner')     # Join two tables on adgroup_id

df3.head()
pd.concat([df[:4], df[10:15]], axis=0)    # concat on axis，axis=0 - on columns，axis=1 - on rows
pd.concat([df[:4], df[10:15]], axis=1)   # on rows it means it will join on index
pd.concat([df[:4], df[10:15].reset_index(drop=True)], axis=1)
df.dropna().shape    # Drop NaN
df.fillna(0).head(10)    
df.fillna(df['brand'].mean()).head(10)    # Use average
df.fillna(method='ffill').head(10)    # Use Previous value
price_des = df['price'].describe()

price_des
valid_max = price_des['50%'] + 3 * (price_des['75%'] - price_des['50%'])

valid_min = price_des['50%'] - 3 * (price_des['75%'] - price_des['50%'])

df['price'] = df['price'].clip(valid_min, valid_max)    # Replace outliers with limits

df['price'].describe()