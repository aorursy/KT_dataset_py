%load_ext autoreload

%autoreload 2

%matplotlib inline



datafolder = "../input"

!ls {datafolder}
import numpy as np 

import pandas as pd



from fastai.tabular.transform import *

from fastai.tabular.data import TabularDataBunch



from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
df_raw = pd.read_csv(f'{datafolder}/train/Train.csv', low_memory=False, parse_dates=['saledate'])

df_raw.shape
# show first 10 rows

display(df_raw.iloc[:10])

# summary statistics of each variable 

display(df_raw.describe(include='all').T)
# Personally I think creating a new variable is better than replacing SalePrice with log(SalePrice) in place,

# because it avoids being transformed by log twice without giving error when the cell is rerun

df_raw['logSalePrice'] = np.log(df_raw['SalePrice'])
# display the datatype and number of NAs of a dataframe

def info(df):

    datatypes = pd.Series(df.dtypes, name='datatype')

    na_count = pd.Series(df.isna().sum(), name='na_count')

    with pd.option_context('display.max_rows',1000,'display.max_columns',1000):

        display(pd.concat([datatypes, na_count],axis=1))

info(df_raw)
# return list of columns of specific datatypes 

def datatype(df):

    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    num_cols = df.select_dtypes(include=['number','bool']).columns.tolist()

    cat_cols = df.select_dtypes(include=['object']).columns.tolist() # may contain other type of data

    print(f"Date columns: {date_cols} \n\nNumerical columns:{num_cols} \n\nString columns: {cat_cols}")

    return date_cols, num_cols, cat_cols

date_cols, num_cols, cat_cols = datatype(df_raw)
# converts datetime dtype to numerical/boolean dtype 

# in the mean time, adds bunch of generic features generated from datetime

add_datepart(df_raw, 'saledate')



# converts string to categorical dtype

cat_to_num = Categorify(cat_cols, num_cols)

cat_to_num(df_raw)
# now all the dtype are acceptable by machine learning models

info(df_raw)
# fill missing values 

fillNA = FillMissing(cat_cols,num_cols)

fillNA(df_raw)
# x: features y: labels 

x = df_raw.drop(['logSalePrice','SalePrice'],axis=1) 

y = df_raw['logSalePrice']
# automate all

preprocessing = [Categorify, FillMissing, Normalize]

data = TabularDataBunch.from_df(f'{datafolder}/train/', df_raw, dep_var='logSalePrice', valid_idx=range(len(df)-2000, len(df_raw)-1),

                                procs=preprocessing, cat_names=cat_cols)
data
#randomforest = RandomForestRegressor(n_jobs=-1) # n_jobs=-1: use all CPUs, n_jobs=1: no parallelism

#randomforest.fit(x, y)