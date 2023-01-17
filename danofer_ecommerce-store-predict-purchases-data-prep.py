import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# define dtypes on loading data  - will speed up and reduce memory use

categorical_dtypes = {

    'event_type':'category', 'product_id':'category',

    'category_id':'category',

       'category_code':'category', 'brand':'category', 

    'user_id':'category', 'user_session':'category'

}
df = pd.read_csv("/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv",dtype=categorical_dtypes) #,nrows=123456

print(df.shape)

print(df.columns)

print(df.dtypes)

df.tail()
## hierarchical code - we can split by digits into up to 3 levels

print(df["category_code"].sample(frac=0.01).nunique())

df["category_code"].value_counts()
df["category_code_level1"] = df["category_code"].str.split(".",expand=True)[0].astype('category')



print(df["category_code_level1"].nunique())

df["category_code_level1"].value_counts()
df.head(9)["category_code"].str.split(".",expand=True)[1]
### setting the dtype on import to categorical doesn't work when concatenating..  + We have a lot of data in this scenario - start with just 1 month

# df = pd.concat([pd.read_csv("/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv",dtype=categorical_dtypes)

#                 ,pd.read_csv("/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Nov.csv",dtype=categorical_dtypes)])



df['event_time'] = pd.to_datetime(df['event_time'],infer_datetime_format=True)





# add joint key for user + product  and/or user + category

# df["user_product"] = (df['user_id'].astype(str)+df['product_id'].astype(str)).astype('category').cat.codes.astype('category')

df["user_category"] = (df['user_id'].astype(str)+df['category_id'].astype(str)).astype('category').cat.codes.astype('category')





## categorical/label encoding of the IDs (instead of string - save memory/file size):

### we still need to define ats categoircal due to the concatenation of the input files

df['user_session'] = df['user_session'].astype('category').cat.codes.astype('category')

df['user_id'] = df['user_id'].astype('category').cat.codes.astype('category')

df['category_id'] = df['category_id'].astype('category').cat.codes.astype('category')

df['product_id'] = df['product_id'].astype('category').cat.codes.astype('category')





print(df.shape)

df.head()
## meory usage

df.info()
df["event_type"].value_counts()
df["product_id"].value_counts().describe()
df.sample(frac=0.02).drop(["event_time"],axis=1).nunique()
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
## set overseved = True to avoid memory error when using categoricals

df_targets["purchased"] = df_targets.groupby(["user_session","product_id"],observed=True,sort=False)["purchased"].transform("max")

df_targets["purchased"].describe()
# keep only rows with the time of addition to cart  = time of prediction

## also, drop duplicates for cases of multiple purchases of same it or readding (only a small amount - a few hundred such cases)

df_targets = df_targets.loc[df_targets["event_type"]=="cart"].drop_duplicates(["user_session","product_id","purchased"])



print(df_targets.shape)

df_targets["purchased"].describe()
df.to_csv("ecom_store_timeSeries.csv.gz",index=False,compression="gzip")



# output a sample of targets for modelling speed

df_targets.drop(["event_type"],axis=1).sample(frac=0.35).to_csv("ecom_store_cart_purchase_labels.csv.gz",index=False,compression="gzip")