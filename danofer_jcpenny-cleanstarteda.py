import pandas as pd

import numpy as np

df = pd.read_csv("../input/jcpenney_com-ecommerce_sample.csv").drop(['uniq_id'],axis=1)

df[['list_price', 'sale_price']] = (df[['list_price', 'sale_price']]

        .applymap(lambda v: str(v)[:4]).dropna().astype(np.float64))
print(df.shape)

df.head()
df.columns
len(set(df.name_title))

df.dropna(subset=['list_price', 'sale_price']).drop_duplicates().shape
df.dropna(subset=['sale_price']).shape
df = df.dropna(subset=['sale_price','list_price']).drop_duplicates(subset=["Reviews","sku","sale_price",'list_price'])

df.shape
len(set(df.name_title))
df["average_product_rating"] = df["average_product_rating"].str.replace(" out of 5","")
df.head()
# Note: this should be used only for metadata /feature engineering, not left in either model directly!

df["sale_discount"] =  100*df['sale_price'].div(df['list_price'])

df["sale_discount"] = df["sale_discount"].apply(lambda v: str(v)[:5]).astype(float)
df.head()
def transform_category_name(category_name):

    try:

        main, sub1, sub2= category_name.split('|')

        return main, sub1, sub2

    except:

        return np.nan, np.nan, np.nan



df['category_main'], df['category_sub1'], df['category_sub2'] = zip(*df['category_tree'].apply(transform_category_name))

df.head()
df.apply(lambda x: len(x.unique()))



## Looks like category isn't the same as the lowest/category_sub2 col!            
df.category_main.value_counts()

# the #2 at the top level is simply NaNs..
df.drop(["category_main",'category_tree'],axis=1,inplace=True)
df.to_csv("jcPenny_subset.csv.gz",index=False,compression="gzip")