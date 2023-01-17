import numpy as np # linear algebra

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.concat([pd.read_csv("/kaggle/input/200-financial-indicators-of-us-stocks-20142018/2016_Financial_Data.csv"),

                     pd.read_csv("/kaggle/input/200-financial-indicators-of-us-stocks-20142018/2015_Financial_Data.csv"),

                     pd.read_csv("/kaggle/input/200-financial-indicators-of-us-stocks-20142018/2014_Financial_Data.csv"),

                     pd.read_csv("/kaggle/input/200-financial-indicators-of-us-stocks-20142018/2017_Financial_Data.csv"),],sort=False).drop_duplicates().sample(frac=1)

print(df_train.shape)

df_train.head()
df_train.columns
df_train.drop(['2017 PRICE VAR [%]', '2016 PRICE VAR [%]', '2015 PRICE VAR [%]', '2018 PRICE VAR [%]'],axis=1,inplace=True)
df_train["Class"].describe()
df_train.to_csv("stock_financials_train.csv.gz",index=False,compression="gzip")