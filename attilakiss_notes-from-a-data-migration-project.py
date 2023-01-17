# the following modules were used in order to carry out the data migration
import sqlalchemy as db  #for establishing postgres connection, general purpose
from sqlalchemy import exc  #Exceptions of the sqlalchemy / useful for 'to_sql' method error logging
import psycopg2  #Python - Postgres connection requirement, postgres specific
import sqlite3  #Python - SQLite connection

import pandas as pd  #data stored pandas dataframes, and bulit-in pandas methods were used for transformation
import numpy as np  #pandas built upon numpy, supplementary module of pandas

import time  #assigning a timestamp for the errorlog file
#Reading the CATEGORICAL tables

#SQLalchemy connection
engine = db.create_engine(postgres_path)  #postgres_path is a string, determined elsewhere
connection = engine.connect()  #establish the connection object with the DB

#variables
df_dict = dict()  #served as an output varibale
df_dict_cat_col_names = dict()  #served as an output variable
tables = ["region", "brand", "model", "category", "car_type", "environmental", "drive"]  #table names for SQL query
messed_up_names = {  
    "category": "kategória",
    "car_type": "kivitel",
    "environmental": "környezetvédelmi",
    "drive": "hajtás"}  #the keys are the names of the tables; the values are the name of the columns in the 'Advertisements' table

#SQL query
for table in tables:
    df_dict[table] = df = pd.read_sql("SELECT * FROM {0};".format(table), connection)  #very simple SELECT statement, table name is replaced with every iteration
    df_dict_cat_col_names[table] = {str(table)+'_id':str(table)+'_name'}  #also generating dictionaries, which are containing the key-value column names in the categorical tables

#replacing column names
for correct_name, messed_name in messed_up_names.items():  #e.g. correct_name = 'category'; messed_name = 'kategória'
                                                           #messed_name can be found in advertisements and catalogs tables
    df_dict[messed_name] = df_dict[correct_name]
    df_dict.pop(correct_name)

    df_dict_cat_col_names[messed_name] = df_dict_cat_col_names[correct_name]
    df_dict_cat_col_names.pop(correct_name)

connection.close()

print('done - populate_categorical_dictionaries()')

#OUTPUTS
# df_dict -> dictionary; categorical tables data can be found here as dataframes (e.g. {'brand':pd.DataFrame(columns=['brand_id','brand_name'])})
# df_dict_cat_col_names -> dictionary; categorical column names can be found here (e.g. {'brand':{'brand_id':'brand_name'}})
#reading the data from the OLD Database

#connecting to the old DB
sql3_database = sqlite_path  #sqlite_path is a string, determined elsewhere
conn_sqlite3 = sqlite3.connect(sql3_database)  #establish the connection object with the DB

#variables
tables = ["advertisements","catalogs"]  #table names for reading
column_names = dict()  #output variable
df_dict_old_tables = dict()  #output variable

#date formats
date_format_advertisements = '%Y%m%d'  #date parsing format1
date_format_catalogs = '%Y'  #date parsing format2

#SQL query
for table in tables:
    df_dict_old_tables[table] = df = pd.read_sql("SELECT * FROM {0};".format(table),  #simple query, table name is replaced with every iteration
    conn_sqlite3,
    parse_dates={
        'agegroup':date_format_advertisements, 
        'documentvalid': date_format_advertisements,
        'upload_date': date_format_advertisements,
        'sales_date': date_format_advertisements,
        'download_date': date_format_advertisements,
        'sales_update_date': date_format_advertisements,
        'start_production': date_format_catalogs,
        'end_production': date_format_catalogs,
        }  #pandas has an in-bulit support for date parsing, all I had to do is provide the format how the dates were stored
    )
    column_names[table] = df_dict_old_tables[table].columns.tolist()  #saving the column names of the SQL DB in a list (old db)
conn_sqlite3.close()

print('done - read_old_db_tables()')

#OUTPUTS
#df_dict_old_tables: query results saved in df, with keys= 'advertisements', 'catalogs'
#column_names: oldDB column names saved in lists, with keys= 'advertisements', 'catalogs'

#reading data from the NEW Database

#SQLalchemy connection
engine = db.create_engine(postgres_path)  #postgres_path is a string, determined elsewhere
connection = engine.connect()  #establish the connection object with the DB

df_dict = dict()  #output variable
column_names = dict()  #output variable
tables = ["advertisements", "catalogs"]
for table in tables:
    df_dict[table] = df = pd.read_sql("SELECT * FROM {0};".format(table), connection) #simple query, table name is replaced with every iteration
    column_names[table] = df_dict[table].columns.tolist() #saving the column names of the SQL DB in a list (new db)
connection.close()

print('done - read_new_db_tables()')

#OUTPUTS
#column_names -> contains all the columns names from the new DB as a list, with keys= 'advertisements', 'catalogs'

#creating the value pairs dicts for the renaming
column_mapping = dict()  #output variable

column_mapping['advertisements'] = dict(zip(
    old_column_names['advertisements'],
    new_column_names['advertisements']))

column_mapping['catalogs'] = dict(zip(
    old_column_names['catalogs'],
    new_column_names['catalogs']))

#OUTPUT
"""
column_mapping -> dictionary, keys= 'advertisements', 'catalogs'; 
                  contains the names of the old and new column name 
                  (e.g.{'advertisement':{'hirkod':'ad_id'}}; hirkod = old name, ad_id = new name )

used later
"""            
#variables (output variable determination)
new_db_tables = dict()

#creating a copy of the DFs that contains the oldDB data, no modification is made yet
for key in df_dict_old_tables.keys():
    new_db_tables[key] = df_dict_old_tables[key].copy()


#merging the oldDB table with the categorical tables, saved in new DF
for key in df_dict_cat_col_names.keys():
    if key in list(new_db_tables['advertisements'].columns.tolist()):
        new_db_tables['advertisements'] = new_db_tables['advertisements'].merge(  #merging the categorical tables to the dataframe on '_name'
            df_dict_cat[key],
            left_on = key,
            right_on=list(df_dict_cat_col_names[key].values())[0]
            ).drop(columns=[key, list(df_dict_cat_col_names[key].values())[0]])  #removing the original value columns
    elif key in list(new_db_tables['catalogs'].columns.tolist()):  #same method for the catalogs table
        new_db_tables['catalogs'] = new_db_tables['catalogs'].merge(
            df_dict_cat[key],
            left_on=key,
            right_on=list(df_dict_cat_col_names[key].values())[0]
            ).drop(columns=[key, list(df_dict_cat_col_names[key].values())[0]])
    
    #increasing the key values of 'clime' by 1
    #otherwise their key value would have been 0, which cause a key error due to primary key constraints (NOTNULL) in SQL DBs
    new_db_tables['advertisements']['clime'] = new_db_tables['advertisements']['clime'] + 1
    
    #remapping the 'gas' column with their key values
    new_db_tables['advertisements']['gas'] = new_db_tables['advertisements']['gas'].replace(to_replace={
        0: 1,
        1: 2,
        2: 3,
        3: 5,  
        5: 6,  #original values are the keys; number 4 is missing from the keys
        6: 7,
        7: 8,
        8: 9,
        9: 11,
        11: 12,
        12: 15,
        15: 16}, inplace = True)

print('done - categorical_data_mapping()')

#OUTPUT
#new_db_tables -> a dictionary, that contains the modified old database with the keys= 'advertisements', 'catalogs' / categorical keys had been inserted (and values removed)
new_db_tables['advertisements'] = new_db_tables['advertisements'].replace(to_replace={  #replacing the values according to the dictionary
    'priv':0,
    'pro':1,
    'OPEN':0,
    'SOLD':1},
    inplace = False).astype({  #changing the dtype
        'sellertype':'bool',
        'eloresorolas':'bool',
        'status':'bool'
    })

print('done - boolean_data_mapping()')

#OUTPUT
#new_db_tables -> a dictionary, with the keys= 'advertisements', 'catalogs' / boolean variables had been remapped
#replacing comma values to dots in CATALOGS
false_data = ['városi', 'országúti', 'vegyes', 'gyorsulás']
for column in false_data:
    new_db_tables['catalogs'][column] = new_db_tables['catalogs'][column].apply(lambda x: x.replace(',', '.'))


print('done - float_data_correction()')

#OUTPUT
#new_db_tables -> a dictionary, with the keys= 'advertisements', 'catalogs' / commas had been replaced in floaters
#determining the appropirate missing data based on the dtype of the existing columns (old db structure), missing value selection based on the new DB's dtypes 
#just a few examples; total array consisting 20+ elements; format: {'column_name': [dtype, 'current_missing_value']}
messed_data = dict()
messed_data['advertisements'] = {'region_id':['integer','null']}
messed_data['catalogs'] = {'category_id': ['integer', 'na']}


for key in messed_data.keys():
    for k, v in messed_data[key].items():
        if v[0] == 'numeric':
            new_db_tables[key][k].replace(v[1:], np.nan, inplace=True)
            new_db_tables[key][k].replace(r'^\s*$', np.nan, regex=True, inplace = True)
        elif v[0] == 'integer':
            new_db_tables[key][k].replace(v[1:], np.nan, inplace=True)
        elif v[0] == 'date':
            new_db_tables[key][k].replace(v[1:], 'NaT', inplace=True)
        elif v[0] == 'text':
            new_db_tables[key][k].replace(v[1:], 'NA', inplace=True)

print('done - missing_data_correction()')

#OUTPUT
#new_db_tables -> a dictionary, with the keys= 'advertisements', 'catalogs' / missing values had been corrected
#renaming the old DB with the new column names
db_tables_renamed_formated = dict()
db_tables_renamed_formated['advertisements'] = new_db_tables['advertisements'].copy().rename(columns=column_mapping['advertisements']).set_index('ad_id')
db_tables_renamed_formated['catalogs'] = new_db_tables['catalogs'].copy().rename(columns=column_mapping['catalogs']).set_index('catalog_url')


print('done - column_renaming()')

#OUTPUT
#db_tables_renamed_formated -> a dictionary, with the keys= 'advertisements', 'catalogs' / column names are matching with the SQL table column names
#the index of the dataset had been set to the ad_id in the advertisements DF, and catalog_url in the catalogs DF, 
#since the SQL import had failed due to the extra index columns of the dataframes
#determining the proper datatypes for the columns 
#just a few examples; total array consisting plenty elements; format: ({'column_name':'dtype'},)
catalog_datatypes = list()
catalog_datatypes = ({'category_id': 'Int64'})
advertisement_datatypes = list()
advertisement_datatypes= ({'region_id': 'Int64'})

#transformation of the data dtypes in the dataframe
for item in catalog_datatypes:
    for column, dtype in item.items():
        if dtype in ['Int64', 'float64']:  #if the value is numeric, it had to be transformed to numeric, different treatment
            db_tables_renamed_formated['catalogs'][column] = pd.to_numeric(db_tables_renamed_formated['catalogs'][column])
        else:
            db_tables_renamed_formated['catalogs'][column] = db_tables_renamed_formated['catalogs'][column].astype(dtype)

for item in advertisement_datatypes:
    for column, dtype in item.items():
        if dtype in ['Int64', 'float64']:  #if the value is numeric, it had to be transformed to numeric, different treatment
            db_tables_renamed_formated['advertisements'][column] = pd.to_numeric(db_tables_renamed_formated['advertisements'][column])
        else:
            db_tables_renamed_formated['advertisements'][column] = db_tables_renamed_formated['advertisements'][column].astype(dtype)


#OUTPUT
#db_tables_renamed_formated -> a dictionary, with the keys= 'advertisements', 'catalogs' / dtypes are aligned with the SQL table data types
#setting up the DB connection
engine = db.create_engine(postgres_path)
connection = engine.connect()

#determining the error log variables
catalog_errors = pd.DataFrame()
catalog_error_types = pd.DataFrame(columns=['catalog_url','error_type'])  #the failed rows will be collected in this dataframe
advertisements_errors = pd.DataFrame()
advertisements_error_types = pd.DataFrame(columns = ['ad_id','error_type'])

#loading the DB into the Postgres SQL DB
if migration_object == 1:  #prompt value from the user
    for i in range(len(db_tables_renamed_formated['catalogs'])):
        try:
            db_tables_renamed_formated['catalogs'].iloc[i:i+1].to_sql(  #all the rows one-by-one (makes it very slow)
                name = 'catalogs', 
                if_exists='append', 
                con=connection)
        except exc.DBAPIError as ex:  #if a row fails, row + reason of failure storing in DF / the psycopg2 exception is wraped in sqlalchemy's 'DBAPIError'
                catalog_errors = catalog_errors.append(db_tables_renamed_formated['catalogs'].iloc[i:i+1])
                catalog_error_types = catalog_error_types.append({
                    'catalog_url': db_tables_renamed_formated['catalogs'].iloc[i:i+1].index[0],  #serving as an ID for merging
                    'error_type': type(ex),
                    'pgerror': ex.orig.pgerror},  #detailed error description also saved
                    ignore_index=True)                
                continue

    catalog_error_types = catalog_error_types.set_index('catalog_url')  #turning the catalog_url as an index
    catalog_errors = catalog_errors.merge(  #merging the reason of failure to the df, merge on indicies!
        right = catalog_error_types,
        how='outer',
        left_index=True,
        right_index=True)
    mig_time = time.strftime("%Y%m%d_%H%M%S")  #getting the timestamp of the migration running
    catalog_errors.to_csv('catalog_errors_{0}.csv'.format(mig_time))  #saving the log dataframe as a .csv