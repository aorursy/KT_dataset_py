import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.concat([pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Oct.csv")

                ,pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv")])



df['event_time'] = pd.to_datetime(df['event_time'],infer_datetime_format=True)



## categorical/label encoding of the session IDs (instead of string - save memory/file size):

df['user_session'] = df['user_session'].astype('category').cat.codes



print(df.shape)

df.head()
df["event_type"].value_counts()
df["product_id"].value_counts().describe()
df.columns
df.drop(["event_time"],axis=1).nunique()
## get all "positive events" , then later we'll add 0s

df_targets = df.loc[df["event_type"].isin(["cart","purchase"])].drop_duplicates(subset=['event_type', 'product_id',

                                                                                        'price', 'user_id',

                                                                                        'user_session'])



print(df_targets.shape)

df_targets.tail()
## not filtering by price (discount?) doesn't change much

df_targets.drop_duplicates(subset=['event_type', 'product_id', 'user_id', 'user_session']).shape[0]
## ## could also do this with np.where  ;  or stack + inindex ; or with a join (on filtered rows containing purhcase and fillna(0))



# # df2["purchase"] = (df2["event_type"]=="purchase").astype(int)



# df2["purchase"] = np.where(df2["event_type"]=="purchase",1,0)



## https://stackoverflow.com/questions/48175172/assign-a-pandas-series-to-a-groupby-operation
## laziest option - add a row where purchased, groupby(max), then keep rows at time of added to cart



# df_targets["purchased"] = (df_targets["event_type"]=="purchase").astype(int) #np.where() ## only kept 1s - bug ..



df_targets["purchased"] = np.where(df_targets["event_type"]=="purchase",1,0)

print(df_targets.shape)

df_targets["purchased"].describe()
df_targets["purchased"] = df_targets.groupby(["user_session","product_id"])["purchased"].transform("max")

df_targets["purchased"].describe()
df_targets.loc[df_targets["event_type"]=="cart"].shape[0]
# keep only rows with the time of addition to cart  = time of prediction

## also, drop duplicates for cases of multiple purchases of same it or readding (only a small amount - a few hundred such cases)

df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","purchased"])



print(df_targets.shape)

df_targets["purchased"].describe()
df.to_csv("ecom_cosmetics_timeSeries.csv.gz",index=False,compression="gzip")

df_targets.to_csv("ecom_cosmetics_cart_purchase_labels.csv.gz",index=False,compression="gzip")