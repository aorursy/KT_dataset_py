## DONE: Imports for any data science project

import operator as opt

import numpy as np 

import pandas as pd 

import os

import gc

from contextlib import contextmanager

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
## DONE: Create files dictionary for any file in the input directory



files = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files[filename] = os.path.join(dirname, filename)

        print(os.path.join(dirname, filename))
## DONE: Loading files in pandas dataframe [1.1]



def load_file_into_dataframe(file_name):

    df = pd.read_csv(files[file_name])

    return df



#     print(f"Dataframe shape:", df.shape)

#     print(f"\nDataframe data type(s):\n",df.dtypes)

#     print(f"\nDataframe first five rows:\n",df.head())
## DONE: Check on Pipeline [1.1]  



train_df = load_file_into_dataframe('Train.csv')

test_df = load_file_into_dataframe('Test.csv')

submission_df = load_file_into_dataframe('Submission.csv')
dict(train_df.count(axis=0))
## DONE: Examine shape of the dataframe [1.3.1], Examine exploratory information of the dataframe [1.3.2]

# Examine statistical data within the dataframe [1.3.3], Examine columnwise datatypes in the dataframe [1.3.4]

# Examine number of unique values in a column in the dataframe [1.3.5]

def dataframe_shape(df): 

    """ Examine shape of the dataframe [1.3.1] """

    return df.shape



def dataframe_split_between_null_and_not_null(df):

    not_null_df = df.dropna()

    null_df = df.drop(not_null_df.index)

    print("Actual dataset ", df.shape)

    print("Null dataset ", null_df.shape)

    print("Not null dataset ", not_null_df.shape)

    

def dataframe_info(df):

    """ Examine exploratory information of the dataframe [1.3.2] """

    return df.info()



def dataframe_description(df):

    """ Examine statistical data within the dataframe [1.3.3] """

    return df.describe().T



def dataframe_columnwise_individuals_data_types(df):

    """ Examine columnwise datatypes in the dataframe [1.3.4] """

    return df.dtypes



def check_datatypes_of_col_number(df):

    """ Unique data types column number """

    return df.dtypes.value_counts()



def unique_value_in_specified_dtype_cols(df, dtype='object'):

    """ Shows number of unique values of specified data type in several columns """

    return df.select_dtypes(dtype).apply(pd.Series.nunique, axis = 0)



def number_of_unique_values_in_col(df, col_name='def_col_name'):

    """ Examine number of unique values in a column in the dataframe [1.3.5] """

    return df[col_name].value_counts()



def correlation_matrix_df(df):

    """ Return data frame correlation matrix """

    return df.corr()  #.sort_values(ascending=False)



def check_null_columnwise(df):

    """ Return column wise none value """

    new_df = df.isnull().sum()

    new_df = new_df[new_df > 0]

    return new_df



## TODO: **Examine and visualize distribution of the dataframe based on traget column(s)**



# def visualize_col_distribution(df, col_name):

#     return df[col_name].astype(int).plot.hist();



## LOOKUP: Concentrate here
print(" -" * 30)

print(dataframe_info(train_df)) # info

print(" -" * 30)

print(" -" * 30)

print(dataframe_description(train_df)) # describe

print(" -" * 30)

print(" -" * 30)

print(dataframe_columnwise_individuals_data_types(train_df)) # columnwise data types

print(" -" * 30)

print(" -" * 30)

print(check_datatypes_of_col_number(train_df)) # which datatype has how many columns

print(" -" * 30)

print(" -" * 30)

print(correlation_matrix_df(train_df)) # correlation matrix

print(" -" * 30)

print(" -" * 30)

print( unique_value_in_specified_dtype_cols(train_df, dtype='object')) # count of class labels of object datatype

print(" -" * 30)

print(" -" * 30)

print(check_null_columnwise(train_df))
## DONE: Examine distribution of the dataframe based on traget column [2]



# def check_distribution(df, col_name='default_col_name'):

#     dist_ = df[col_name].value_counts()

#     return dist_



# def viz_distribution(df, col_name='default_col_name'):

#     return df[col_name].plot.hist();



# print(check_distribution(train_df, 'Outlet_Identifier'))

# viz_distribution(train_df, 'Outlet_Identifier')
## TODO: Finding and visualizing outliers using cross tables and pivot table [1.4]
## TODO: Find anomalies within the dataframe checking every column [2]
## DONE: Examine missing values with columnwise statistics in the dataframe [3]



def missing_values_stat_in_columns(df):

    """ Showing a dataframe columnwise missing values in number and percentage of total values """

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

    return mis_val_table_ren_columns



# def missing_values_dataframe_in_series(df):

#     """ Missing values count in the dataframe as series """

#     r_df = df.isnull().sum().sort_values(ascending=False)

#     df = r_df[r_df.iloc[:] != 0]

#     return df



def drop_columns_from_dataframe(df, col_list= ['colA','colB']):

    """ Drops a list of columns with provided column names """

    return df.drop(col_list, axis=1, inplace=True)



def generate_condition(df,col_name='colA', value= 'valA', com_op = opt.lt):

    """Return row indices based on column name, column value and condition operator i.e ['eq', 'ne', 'ge', 'le', 'gt', 'lt']"""

    dtype_ = df[col_name].dtype

    value_ = pd.Series([value], dtype=dtype_)

    indices = df[com_op(df[col_name], value_[0])].index

    return indices



def drop_rows_based_on_indices(df, indices = None):

    """ Return dataframe and drop rows based on given row indices """

    return df.drop(indices, inplace = True)



def missing_value_manipulation(df, missing_percentage = 30.0):

    """ Return dataframe and drop columns based on their missing data percentage """

    missing_percentage_df = missing_values_stat_in_columns(df)

    cols_to_drop = set()

    

    for index, row in missing_percentage_df.iterrows():

        if row['% of Total Values'] >= float(missing_percentage):

            cols_to_drop.add(index)

    print(cols_to_drop)

    return cols_to_drop # list of row index to drop



def check_correlation_against_target_column(df, col_name, target_col_name):

    """ Return correlation value of a column against target column """

    return df[col_name].corr(df[target_col_name])



def correaltion_dataframe(df, target_col, positive= True):

    """ Return correlation dataframe based on positive and negative """

    correlations =  df.corr()[target_col].sort_values(ascending=False)

#     return correlations

    if positive == True:

        return correlations[correlations > 0]

    else:

        return correlations[correlations < 0]

    

## 1. Columnwise missing value check [done]

## 2. Missing value percentage check [done]

## 3. Addition or deletion of column based on missing value threshold i.e percentage [done]

## 4. Missing value column significance based on correlation() [done]

## 5. numerical -> (MOD -> checking on highest frequency | if std() is very small in that sense we can impute mean()) | median(), categorical -> (maximum frequency)



## TODO: 
correaltion_dataframe(train_df, 'Item_Outlet_Sales', positive= False)
# train_df.head()

# generate_condition(df,col_name='colA', value= 'valA', com_op = opt.lt)

generate_condition(train_df,'Outlet_Establishment_Year','1999')
# missing_values_stat_in_columns(train_df)

missing_value_manipulation(train_df, 10.0)
## DONE: Encode categorical variables in the dataframe i.e label encoding, one-hot encoding



def label_encoding(df_train, df_test): ## TODO: add df_test dataframe

    le = LabelEncoder()

    le_count = 0

    

    for col in df_train:

        if df_train[col].dtype == 'object' and len(list(df_train[col].unique())) <= 2:

            le.fit(df_train[col])

            df_train[col] = le.transform(df_train[col])

            df_test[col] = le.transform(df_test[col]) ## TODO: while test dataframe will be provided

            le_count += 1

    print('%d columns were label encoded.' % le_count)

        

def one_hot_encoding(df_train, df_test): ## TODO: add df_test dataframe

    df_train = pd.get_dummies(df_train)

    df_test = pd.get_dummies(df_test) ## TODO: while test dataframe will be provided



    print('Training Features shape: ', df_train.shape)

    print('Testing Features shape: ', df_test.shape) ## TODO: while test dataframe will be provided

    

    return df_train, df_test



# label_encoding(df_train, df_test)

# df_train, df_test = one_hot_encoding(df_train, df_test)
## DONE: Align dataframe based on primary and foreign key of two dataframe



def align_dataframe(df_train, df_test, target_col):

    target_labels = df_train[target_col]

    df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

    df_train[target_col] = target_labels

    

    print('Training Features shape: ', df_train.shape)

    print('Testing Features shape: ', df_test.shape)

    

# align_dataframe(df_train, df_test, 'TARGET')