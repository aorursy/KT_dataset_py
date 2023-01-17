import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline 

import math


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#LOAD DATA
print("Loading data...")
train = pd.read_table("../input/train.tsv")
test = pd.read_table("../input/test.tsv")
print(train.shape)
print(test.shape)
train["brand_name"] = train["brand_name"].fillna("missing")
test.brand_name = test.brand_name.fillna("missing")


train["category_name"] = train["category_name"].fillna("missing")
test.category_name = test.category_name.fillna("missing")
cond_g = train.groupby("item_condition_id")
cat_g = train.groupby("category_name")
brand_g = train.groupby("brand_name")
ship_g = train.groupby("shipping")

cond_price = cond_g.price.mean()
cat_price = cat_g.price.mean()
brand_price = brand_g.price.mean()
ship_price = ship_g.price.mean()
# train.groupby(by=["item_condition_id", "category_name", "brand_name", "shipping"])
gg = train.groupby(by=["item_condition_id", "category_name", "brand_name", "shipping"])
pp = gg["price"].mean()
tp = {}
for cond in test.item_condition_id.unique():
    data = pp[cond]
    t_cond = test[test.item_condition_id==cond]
    
    for cat in test.category_name.unique():
        
        if cat not in pp[cond].index:
            continue
        cat_data = data[cat]
        t_cat = t_cond[t_cond.category_name==cat]
        
        for brand in t_cat.brand_name.unique():
            if brand not in pp[cond][cat].index:
                continue
            b_data = cat_data[brand]
            t_brand = t_cat[t_cat.brand_name==brand]
            for index, row in t_brand.iterrows():
                b = cat_data[row.brand_name]
                if row.shipping not in b.index:
                    tp[index] = b.mean()
                else:
                    tp[index] = b[row.shipping].mean()
        print(cat)
        
    print("cond:", cond)
out = pd.DataFrame([pd.Series(tp)]).T
out.columns = ["price"]
out["test_id"] = out.index
out = out[["test_id", "price"]]
out.to_csv("yyallday.csv",index=None)
# out = []
# count = 0
# for index, row in test.iterrows():
#     df_cond = pp[row.item_condition_id]

#     if row.category_name not in df_cond.index:
#         o = df_cond.mean()
#     else:
#         df_cat = df_cond[row.category_name]
#         if row.brand_name not in df_cat.index:
#             o = df_cat.mean()
            
#         else:
#             df_brand = df_cat[row.brand_name]
#             if row.shipping not in df_brand.index:
#                 o = df_brand.mean()
#             else:
#                 o = df_brand[row.shipping].mean()
                    
#     out.append(o)
#     count += 1
#     if count %1000 == 0:
#         print(count)
# out = []
# count = 0
# for index, row in test.iterrows():
#     df_cond = pp[row.item_condition_id]

#     if row.category_name not in df_cond.index:
#         o = df_cond.mean()
#     else:
#         df_cat = df_cond[row.category_name]
#         if row.brand_name not in df_cat.index:
#             o = df_cat.mean()
            
#         else:
#             df_brand = df_cat[row.brand_name]
#             if row.shipping not in df_brand.index:
#                 o = df_brand.mean()
#             else:
#                 o = df_brand[row.shipping].mean()
                    
#     out.append(o)
#     count += 1
#     if count %1000 == 0:
#         print(count)
# out = []
# count = 0
# for index, row in test.iterrows():
    
#     cond = row["item_condition_id"]
#     cat = row["category_name"]
#     brand = row["brand_name"]
#     ship = row["shipping"]
    
#     o = 0
    
#     if cond in list(cond_price.index):
#         o += cond_price[cond]
#     elif cond in list(cond_price.index):
#         o += cat_price[cat]
#     elif cond in list(cond_price.index):
#         o += brand_price[brand]
#     elif cond in list(cond_price.index):
#         o += ship_price[ship]
    
#     o = o/4
    
#     out.append(o)
#     count += 1
#     if count%1000==0:
#         print(count)
    
# output = pd.DataFrame([pd.Series(out)]).T
# output["test_id"] = output.index
# output.columns = ["price", "test_id"]
# output = output[["test_id", "price"]]
# output
# output.to_csv("test.csv", index=None)