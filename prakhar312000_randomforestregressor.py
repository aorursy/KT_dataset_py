# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import forest

from sklearn.tree import export_graphviz
df_raw = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv" , low_memory=False   , parse_dates=["saledate"])
df_raw.head(5)
df_raw.saledate
df_raw["sale_year"] = df_raw.saledate.dt.year

df_raw["sale_date"] = df_raw.saledate.dt.day

df_raw["sale_month"] = df_raw.saledate.dt.month
df_raw1 = df_raw.drop('saledate',axis = 1)
df_raw1.head(5)
from pandas.api.types import is_string_dtype,is_numeric_dtype
def train_cats(df):

    for l,c in df.items():

        df[l] = c.astype('category').cat.as_ordered()
train_cats(df_raw1)
df_raw.SalePrice = np.log(df_raw.SalePrice)
from sklearn.ensemble import RandomForestRegressor
df_raw1.isnull().sum().sort_index()/len(df_raw)
def scale_vars(df, mapper):

    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper
def get_sample(df,n):

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()
def numericalize(df, col, name, max_n_cat):

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1
def fix_missing(df,col,name,na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            if name in na_dict:

                filler = na_dict[name]

            else:

                col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict        

            

            

                

            

            
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None, 

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    

    if not ignore_flds:

        ignore_flds=[]

    if not skip_flds:

        skip_flds=[]

    if subset:

        df = get_sample(df,subset)

    else:

        df = df.copy()

    ignored_flds = df.loc[:,ignore_flds]

    df.drop(ignore_flds,axis = 1, inplace = True)

    if preproc_fn:

        preproc_fn(df)

    if y_fld is None:

        y = None

    else:

        if not is_numeric_dtype(df[y_fld]):

            df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds , axis=1,inplace = True)

    

    if na_dict is None:

        na_dict={}

    else:

        na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for l,c in df.items():

        na_dict = fix_missing(df,c,l,na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis = 1, inplace = True)

    if do_scale:

        mapper = scale_vars(df,mapper)

    for l,c in df.items():

        numericalize(df,c,l,max_n_cat)

    df = pd.get_dummies(df,dummy_na = True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df,y,na_dict]

    if do_scale:

        res = res+[mapper]

    return res    

        
df , y , nas = proc_df(df_raw1, 'SalePrice')
m = RandomForestRegressor(n_jobs=-1)

m.fit(df,y)

m.score(df,y)
def split_vals(a,n):

    return a[:n].copy(), a[n:].copy()

n_valid = 12000

n_trn = len(df) - n_valid

raw_train, raw_valid  = split_vals(df_raw1, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)

X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
import math

def rmse(x,y):

    return math.sqrt(((x-y)**2).mean())

def print_score(m):

    res = [rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid) , y_valid)]
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)