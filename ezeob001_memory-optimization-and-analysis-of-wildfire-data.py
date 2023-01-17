# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import matplotlib.ticker as ticker

import matplotlib.dates as mdates

from mpl_toolkits.basemap import Basemap

import hashlib

import enum

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#Enum class to specify two file format(csv and parquet)

class FileType(enum.Enum):

    csv_format = 1

    parquet_format = 2
def extract_path_name(dirname_path='/kaggle/input') -> str:

    path_for_database = ""

    for dirname, _, filenames in os.walk(dirname_path):

        for filename in filenames:

            path_for_database = os.path.join(dirname, filename)

    return path_for_database



#Connect to database file_type:FileType

def open_connection():

    path_for_database = extract_path_name()

    conn = sqlite3.connect(path_for_database)

    return conn





def get_all_tables_in_database(conn=None):

    

    new_conn = False

    

    if conn is None:

        conn = open_connection()    

        new_conn = True

    cursor = conn.cursor()

    result = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

    cursor.close()

    

    if new_conn:

        conn.close()

    

    return sorted(list(zip(*result))[0])  # Note: * makes it iteritable



conn = open_connection()

all_tables = get_all_tables_in_database(conn)

conn.close()

#Get the fire database into RAM

#To preventing query the database all the time we cache the data



def read_from_database(sql_query, conn, parameters):

    df_raw = pd.DataFrame()

    try:

        print("Reading from database")

        df_raw = pd.read_sql_query(sql_query, con=conn, params=parameters)

    except (KeyboardInterrupt, SystemExit):

        conn.close()

    return df_raw



def create_cache_folder(name):

    if not os.path.isdir(name):

        print("making _cache directory")

        os.makedirs(name)





def execute_sql_query(sql_query,file_type: FileType, parameters=None, conn=None):

    """

    Method to query data from SQL database and return panda dataframe

    

    Parameters

    

    sql_query : str

    parameters : dict

    FileType : csv or parquet format

    """

    new_conn = False

    if conn is None:

        conn = open_connection()

        new_conn = True

    #Hash the query

    query_hash = hashlib.sha1(sql_query.format(parameters).encode()).hexdigest()

    #Create the filepath

      

    if file_type is FileType.csv_format:

        file_path_csv = os.path.join("_cache","{}.csv".format(query_hash))

        if os.path.exists(file_path_csv):

            print("Reading from csv file")

            df_raw = pd.read_csv(file_path_csv)

        else:

            df_raw = read_from_database(sql_query, conn, parameters)

            create_cache_folder("_cache")

            print("writing dataframe to a csv file")

            df_raw.to_csv(file_path_csv, index=False)

    else: #It is parquet format

        file_path_parquet = os.path.join("_cache","{}.parquet".format(query_hash))

        if os.path.exists(file_path_parquet):

            print("Reading from parquet file")

            df_raw = pd.read_parquet(file_path_parquet)

        else:

            df_raw = read_from_database(sql_query, conn, parameters)

            create_cache_folder("_cache")

            print("writing dataframe to a parquet file")

            df_raw.to_parquet(file_path_parquet)    

            

    if new_conn:

        conn.close()

    

    return df_raw



sql_query = "SELECT * FROM Fires;"  

df = execute_sql_query(sql_query, FileType.csv_format)



        
#Lets understand the memory usage for the data more

df.info(memory_usage='deep')
#understand the limits for df data

df.describe()


for dtype in ['float','int','object']:

    selected_dtype = df.select_dtypes(include=[dtype])

    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()

    mean_usage_mb = mean_usage_b / 1024 ** 2

    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
def mem_usage(pandas_obj):

    if isinstance(pandas_obj, pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2

    return "{:03.2f} MB".format(usage_mb)

#Optimize memory usage for int values

df_int = df.select_dtypes(include=['int'])

print(df_int.describe())

print("The number of NULLS in int are {} ".format(df_int.isnull().values.sum()))

print("==========================================================")

converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')

print(converted_int.describe())

print("==========================================================")

print("Memory usage for int value before conversion: {}".format(mem_usage(df_int)))

print("Memory usage for int value after conversion: {}".format(mem_usage(converted_int)))

compare_ints = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)

compare_ints.columns = ['before','after']

compare_ints.apply(pd.Series.value_counts)
df_float = df.select_dtypes(include=['float'])

print(df_float.describe())

print("The number of NULLS in float are {} ".format(df_float.isnull().values.sum()))

print("==========================================================")

converted_float = df_float.apply(pd.to_numeric,downcast='float')

print(converted_float.describe())

print("==========================================================")

print("Memory usage for float value before conversion: {}".format(mem_usage(df_float)))

print("Memory usage for float value before conversion: {}".format(mem_usage(converted_float)))

compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes],axis=1)

compare_floats.columns = ['before','after']

compare_floats.apply(pd.Series.value_counts)
#MSE for 

print("MSE for converted and original values for DISCOVERY_DATE {} ".format(np.square(df_float.DISCOVERY_DATE-converted_float.DISCOVERY_DATE).mean()))

print("MSE for converted and original values for CONT_DATE {} ".format(np.square(df_float.CONT_DATE-converted_float.CONT_DATE).mean()))
optimized_df = df.copy()

optimized_df[converted_int.columns] = converted_int

optimized_df[converted_float.columns] = converted_float

print("Memory usage before int and float reduction: {}".format(mem_usage(df)))

print("Memory usage after int and float reduction: {}".format(mem_usage(optimized_df)))
df_obj = df.select_dtypes(include=['object']).copy()

print(df_obj.describe())

print(df_obj.shape)

print("The number of NULLS in object/categoric are {} ".format(df_float.isnull().values.sum()))
dow = df_obj.SOURCE_SYSTEM_TYPE

print(dow.head())

dow_cat = dow.astype('category')

print(dow_cat.head())
#To see the converted category attributes

dow_cat.cat.codes.head()
print("Memory usage for the SOURCE_SYSTEM_TYPE object type before conversion is {}".format(mem_usage(dow)))

print("Memory usage for the SOURCE_SYSTEM_TYPE object type after conversion is {}".format(mem_usage(dow_cat)))
converted_obj = df_obj.copy()

for col in df_obj.columns:

    num_unique_values = len(df_obj[col].unique())

    num_total_values = len(df_obj[col])

    if num_unique_values / num_total_values < 0.5:

        converted_obj.loc[:,col] = df_obj[col].astype('category')
print("Memory usage for the object type before conversion is {}".format(mem_usage(df_obj)))

print("Memory usage for the object type after conversion is {}".format(mem_usage(converted_obj)))

compare_obj = pd.concat([df_obj.dtypes,converted_obj.dtypes],axis=1)

compare_obj.columns = ['before','after']

compare_obj.apply(pd.Series.value_counts)
optimized_df[converted_obj.columns] = converted_obj

print("Memory usage for the before conversion is {}".format(mem_usage(df)))

print("Memory usage after applying all conversion unit is {}".format(mem_usage(optimized_df)))
optimized_df.describe()
#df.info()

optimized_df.info(memory_usage='deep')
optimized_df['FIRE_YEAR'].head()
plt.figure(figsize=(16, 9))

sns.set(style="white")

ax = sns.countplot(x="FIRE_YEAR", data = optimized_df, palette="Blues_d")

ax.set_title("Frequency of wildfire per Year", fontdict = {'fontsize':30, 'fontweight':'bold'})

ax.set_xlabel("Year", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.set_ylabel("Frequency", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.grid(which = 'major',color = 'grey', linewidth = 0.2)
labels = {  'A' : '(0-0.25]',

            'B' : '[0.26-9.9]',

            'C':'[10.0-99.9]', 

            'D':'[100-299]', 

            'E':'[300-999]', 

            'F':'[1000-4999]', 

            'G': '[5000 - inf)'}



plt.figure(figsize=(16, 9))

ax = sns.countplot(x="FIRE_SIZE_CLASS", data = optimized_df, palette="Blues_d")

ax.set_title("Count of Fire by Size", fontdict = {'fontsize':30, 'fontweight':'bold'})

ax.set_xlabel("Classes of Wildfires(Acres)", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.set_ylabel("Frequency", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.set_xticklabels(labels.values())

ax.grid(which = 'major',color = 'grey', linewidth = 0.2)

plt.figure(figsize=(16, 9))

ax = sns.countplot(x='STATE', data=optimized_df, palette='Blues_d', order = optimized_df.STATE.value_counts().index)

ax.set_title("WildFire by states", fontdict = {'fontsize':30, 'fontweight':'bold'})

ax.set_xlabel("States", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.set_ylabel("Frequency", fontdict = {'fontsize':20, 'fontweight': 'medium'})
plt.figure(figsize=(16, 9))

ax = sns.FacetGrid(col = 'FIRE_YEAR', height = 5, aspect = 2, col_wrap=4, data=optimized_df)

ax.map(sns.countplot, 'STATE', order = optimized_df.STATE.unique())

for i in ax.axes.flat:

    i.set_title(i.get_title(), fontsize='xx-large')

    i.set_ylabel(i.get_ylabel(), fontsize='xx-large')

    i.set_xlabel(i.get_xlabel(),fontsize = 'xx-large')
plt.figure(figsize=(16, 9))

fire_date = optimized_df.groupby('DISCOVERY_DOY').size()

#print(fire_date.values)

ax = sns.scatterplot(data=fire_date, s=150)

ax.grid(which = 'major',color = 'grey', linewidth = 0.05)

ax.set_title("Frequency of wildfire by Day of the year", fontdict = {'fontsize':30, 'fontweight':'bold'})

ax.set_xlabel("Days in a year", fontdict = {'fontsize':20, 'fontweight': 'medium'})

ax.set_ylabel("Frequency", fontdict = {'fontsize':20, 'fontweight': 'medium'})
sns.heatmap(pd.crosstab(df.FIRE_YEAR, optimized_df.STAT_CAUSE_DESCR))
sql_query = "SELECT * FROM Fires;"  

df2 = execute_sql_query(sql_query, FileType.parquet_format)
def check_file_size(file_name):

    if os.path.exists(file_name):

        statinfo = os.stat(file_name)

        print("The size of the {} is {} bytes".format(file_name,statinfo.st_size))

    else:

        print("File not in the directory")

        return None

    return statinfo.st_size



sql_query = "SELECT * FROM Fires;"

parameters=None

query_hash = hashlib.sha1(sql_query.format(parameters).encode()).hexdigest()

file_path_csv = os.path.join("_cache","{}.csv".format(query_hash))

file_path_parquet = os.path.join("_cache","{}.parquet".format(query_hash))   

csv = check_file_size(file_path_csv)

parquet = check_file_size(file_path_parquet)

print("The differece between csv - {} bytes and parquet - {} bytes is {} and the percentage ratio of parquet to csv is {}%".format(csv, parquet, (csv-parquet),(parquet/csv) *100 ) )

import pyarrow.parquet as pq

parquet_file = pq.ParquetFile(file_path_parquet)

print(parquet_file.metadata)

print("=======================================================")

print(parquet_file.schema)
parquet_file.read_row_group(0)