import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
train = pd.read_csv("../input/train.csv")
history = pd.read_csv("../input/historical_user_logs.csv", nrows= 40_00_000)
display(train.head())
display(history.head())
def datatypes_insight(data):
    display(data.dtypes.to_frame())
    data.dtypes.value_counts().plot(kind="barh")
datatypes_insight(train)
datatypes_insight(history)
train.shape,history.shape
df_train = train.merge(history, on = "user_id")
df_train.shape
df_train.isnull().sum().plot(kind="barh")
df_train.columns
df_train.drop(['session_id','user_id','product_category_2'], axis=1, inplace=True)
# yourdf.drop(['columnheading1', 'columnheading2'], axis=1, inplace=True)
df_train['DateTime_x'].dtype
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

df_train = mis_impute(df_train)
df_train.isnull().sum()
df_train.to_csv("new_train.csv",index = False)