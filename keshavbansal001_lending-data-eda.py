# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(np.NaN)

#df2 = pd.read_excel('/kaggle/input/lending-club-loan-data/LCDataDictionary.xlsx')
#df3 = pd.read_excel('/kaggle/input/lending-club-loan-data/LCDataDictionary.xlsx',sheet_name='RejectStats')
#df4 = pd.read_excel('/kaggle/input/lending-club-loan-data/LCDataDictionary.xlsx',sheet_name='browseNotes')
#df2.info(),df3.info(),df4.info()

file = '/kaggle/input/lending-club-loan-data/loan.csv'
#df.shape
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = np.round(df[col], 3)
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = np.round(df[col], 2)
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, na_values=['N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null',np.inf,np.NaN,0,0.0])
    return df
df = import_data(file)

def memory_diff(df,func):
    start_mem = df.memory_usage().sum() / 1024**2
    df = func(df)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
df.info()
list_of_floats = []
list_of_objects = []
list_of_int = []
col_del_ = []
#list_of_dtype = df.dtypes.unique()
len_df = df.shape[0]
df.shape
list_of_columns = df.columns
arr = np.array([19,47,55,112,123,124,125,128,129,130,133,139,140,141]) -1

df.drop(list_of_columns[arr],axis=1,inplace=True)
df.shape
list_of_columns = df.columns
list_of_dtype = df.dtypes.unique()
list_of_dtype = [str(x) for x in list_of_dtype]
def check_null(ls):
    col_del = []
    for _ in ls:
        x = df[_].isnull().sum()
        if x >= 2250000:
            col_del.append(_)
            print(_,x)
    return col_del
'''
def unnecessary_columns(ls):
    unn_col = []
    for _ in ls:
        x = df[_].isnull().sum()
        if x >= 22500000:
            unn_col.append([_,x])
            print(_,x)
    return unn_col
'''
def return_list(dtpe,ls = []):
    for x in df.columns: 
        try:
            if df[x].dtype == dtpe:
                ls.append(x)
        except Exception as e:
            print(str(e))
            print(df[x].dtype)
            print(x)
            #if df[x].dtype == 'float64':
            #    ls.extend(x)
    return ls
def column_deletion(df):
    # seperate the datatypes
    dtype_list = []
    for each in range(len(list_of_dtype)):
        dtype_list.append(return_list(list_of_dtype[each],[]))
    '''list_of_floats = return_list(list_of_dtype[0],[])
    list_of_objects = return_list(list_of_dtype[2],[])
    list_of_int = return_list(list_of_dtype[1],[])
    '''
    print(list_of_dtype)
    print(dtype_list)
    col_null_ = []
    for each in range(len(dtype_list)):
    # check the null values
        col_null_.extend(check_null(dtype_list[each]))

    '''# check unnecessary columns
    unn_col_ = unnecessary_columns(list_of_floats)
    unn_col_.append(check_null(list_of_int))
    unn_col_.append(check_null(list_of_objects))'''
    
    # delete unnecessary columns
    print(col_null_)
    df.drop(col_null_,axis=1,inplace=True)
    print(df.shape)
    
    return df
    
df = column_deletion(df)
df.shape
df = reduce_mem_usage(df)
list_of_columns = []
for each in df.columns:
    list_of_columns.append(each)
    print(each)
ls = []
for _ in list_of_columns:
    x = df[_].isnull().sum()
    ls.append([_,x])
    print(_,x)
ls
ls = pd.DataFrame(ls,columns=['Feature','Null_count'])
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
ls = ls.sort_values(by='Null_count', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
list_of_dtype = df.dtypes.unique()
list_of_dtype = [str(x) for x in list_of_dtype]
list_of_dtype
list_of_floats = []
list_of_int32 = []
list_of_int64 = []
list_of_objects = []
for x in df.columns: 
        try:
            if df[x].dtype == 'int32':
                list_of_int32.append(x)
            elif df[x].dtype == 'int64':
                list_of_int64.append(x)
            elif df[x].dtype == 'object':
                list_of_objects.append(x)
            elif df[x].dtype == 'float64':
                list_of_floats.append(x)
        except Exception as e:
            print(str(e))
            print(df[x].dtype)
            print(x)
list_of_floats
def check_corr(ls):
    cor_ls = []
    for each in ls:
        for _ in ls:
            cor_ls.appen(df[each].corr(df[_]))
    return cor_ls
corr_list_floats = df[list_of_floats].corr()
corr_list_floats
corr_list_int32 = df[list_of_int32].corr()
corr_list_int64 = df[list_of_int64].corr()
type(corr_list_int32)
corr_list_int32.to_csv("int_32_corr.csv")
corr_list_int64.to_csv("int_64_corr.csv")
corr_list_floats.to_csv("floats_corr.csv")
df.columns
list_of_floats
#list_of_int32
#list_of_int64
#list_of_objects
list_of_int32
list_of_int64
list_of_objects
df.to_csv("adjusted_df.csv")
start_mem = df.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
len(df)
df.shape
import matplotlib.pyplot as plt
import seaborn as sns
corr = corr_list_int32
ax = sns.heatmap(
    corr,annot=True, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
corr = corr_list_int64
ax = sns.heatmap(
    corr,annot=True, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
from matplotlib.pyplot import figure
corr = corr_list_floats
figure(figsize=(30,30))
ax = sns.heatmap(
    corr,annot=True, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
