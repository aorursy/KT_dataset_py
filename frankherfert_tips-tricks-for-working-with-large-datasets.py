import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

from sklearn import preprocessing
import random
%%time
train = pd.read_csv("../input/train.csv")
train.info(memory_usage="deep")
train.memory_usage(deep=True)
train.memory_usage(deep=True) * 1e-6
train.memory_usage(deep=True).sum() * 1e-6
print("size before:", train["activation_date"].memory_usage(deep=True) * 1e-6)
train["activation_date"] = pd.to_datetime(train["activation_date"])
print("size after: ", train["activation_date"].memory_usage(deep=True) * 1e-6)
print("size before:", train["region"].memory_usage(deep=True) * 1e-6)
train["region"] = train["region"].astype("category")
print("size after :", train["region"].memory_usage(deep=True) * 1e-6)
print("size before:", train["city"].memory_usage(deep=True) * 1e-6)
train["city"] = train["city"].astype("category")
print("size after :", train["city"].memory_usage(deep=True) * 1e-6)
def convert_columns_to_catg(df, column_list):
    for col in column_list:
        print("converting", col.ljust(30), "size: ", round(df[col].memory_usage(deep=True)*1e-6,2), end="\t")
        df[col] = df[col].astype("category")
        print("->\t", round(df[col].memory_usage(deep=True)*1e-6,2))
convert_columns_to_catg(train, column_list=["param_1", "param_2", "param_3", "parent_category_name", "category_name", "user_type"])
print(train.memory_usage(deep=True)*1e-6)
print("total:", train.memory_usage(deep=True).sum()*1e-6)
train.to_pickle("train.pkl")
# size is shown in bytes again and needs to be converted to megabytes
print("train.csv:", os.stat('../input/train.csv').st_size * 1e-6)
print("train.pkl:", os.stat('train.pkl').st_size * 1e-6)
del train
%%time
train = pd.read_pickle("train.pkl")
# We will remove the file from the Kernels virtual environment.
os.remove("train.pkl")
import gc
import psutil
print("available RAM:", psutil.virtual_memory())

gc.collect()

print("available RAM:", psutil.virtual_memory())
train["region"].value_counts()
train["user_id"].value_counts().head(5)
train["user_id"].value_counts().tail(5)
def create_label_encoding_with_min_count(df, column, min_count=50):
    column_counts = df.groupby([column])[column].transform("count").astype(int)
    column_values = np.where(column_counts >= min_count, df[column], "")
    train[column+"_label"] = preprocessing.LabelEncoder().fit_transform(column_values)
    
    return df[column+"_label"]
train["user_id_label"] = create_label_encoding_with_min_count(train, "user_id", min_count=50)
print("number of unique users      :", len(train["user_id"].unique()))
print("number of unique user labels:", len(train["user_id_label"].unique()))
train.loc[train["city"]=="Светлый", "region"].value_counts().head()
%%time
train["region_city"] = train.loc[:, ["region", "city"]].apply(lambda s: " ".join(s), axis=1)
print("unique:", len(train["region_city"].unique()))
print("size:  ", train["region_city"].memory_usage(deep=True)*1e-6)
%%time
train["region_city_2"] = train.groupby(["region", "city"])["region"].transform(lambda x: random.random())
print("unique:", len(train["region_city_2"].unique()))
print("size:  ", train["region_city_2"].memory_usage(deep=True)*1e-6)
train["region_city_2_label"] = create_label_encoding_with_min_count(train, "region_city_2", min_count=50)
gc.collect()
train["description_len"] = train["description"].fillna("").apply(len)
train["description_count_words"] = train["description"].fillna("").apply(lambda s: len(s.split(" ")))
train.loc[:, ["user_id_label", "region_city_2_label", "description_len", "description_count_words"]
         ].info()
for col in ["user_id_label", "region_city_2_label", "description_len", "description_count_words"]:
    print(col.ljust(30), "min:", train[col].min(), "  max:", train[col].max())
train.loc[:, ["user_id_label", "region_city_2_label", "description_len", "description_count_words"]
         ].memory_usage(deep=True)*1e-6
train["user_id_label"] = pd.to_numeric(train["user_id_label"], downcast="integer")
train.loc[:, ["user_id_label", "region_city_2_label", "description_len", "description_count_words"]
         ].info()
# note the int16 here
train.loc[:, ["user_id_label", "region_city_2_label", "description_len", "description_count_words"]
         ].memory_usage(deep=True)*1e-6
def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)
        
    if len(list_of_columns)>=1:
        max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing
        print("downcasting integers for:", list_of_columns, "\n")
        
        for col in list_of_columns:
            print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],
                  "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")
            df[col] = pd.to_numeric(df[col], downcast="integer")
            print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))
    else:
        print("no columns to downcast")
    
    gc.collect()
    
    print("done")
downcast_df_int_columns(train)
