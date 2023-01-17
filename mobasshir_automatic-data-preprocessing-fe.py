import time

import operator as opt

import numpy as np 

import pandas as pd 

import os

import gc

from contextlib import contextmanager

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import featuretools as ft

import warnings

warnings.filterwarnings('ignore')
#### Function definition ####

@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



def read_file_names(path='/kaggle/input'):

    files = {}

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            files[filename] = os.path.join(dirname, filename)

            print(filename, ' : ', os.path.join(dirname, filename))

    return files



def load_file_into_dataframe(file_path):

    df = pd.read_csv(file_path)

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # best practice

    df = df.replace('?', np.nan) # best practice

    return df



def dataframe_split_between_null_and_not_null(df):

    not_null_df = df.dropna()

    null_df = df.drop(not_null_df.index)

    percentage = 100 * null_df.shape[0] / df.shape[0]

    print("Actual dataset ", df.shape)

    print("Null dataset ", null_df.shape)

    print("Not null dataset ", not_null_df.shape)

    print(f'Null percentage {percentage}%')

    return df, null_df, not_null_df,percentage



def column_missing_state(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * mis_val / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(1, ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table.shape[0]) +

              " columns that have missing values.")

    return mis_val_table



def column_dropper(df, threshold=.7): # TODO: Shoul be used for generating column list to drop

    missing_percentage_df = column_missing_state(df)

    cols_to_drop = set()

    for index, row in missing_percentage_df.iterrows():

        if row[1] > 100 * threshold:

            cols_to_drop.add(index)

    percentage = 100 * len(cols_to_drop) / df.shape[1]

    print("Actual dataset ", df.shape)

    df = df.drop(list(cols_to_drop), axis=1)

    print("New dataset ", df.shape)

    print(f'Columns drop percentage {percentage}%')

    return df, percentage



def row_dropper(df, threshold =.7):

    col_num = df.shape[1]

    nan_vals = dict(df.count(axis=1)) # How much column every row has

    nan_vals = {key: value for (key, value) in nan_vals.items() if (value / col_num) > threshold}

    percentage = 100 * len(nan_vals.keys()) / df.shape[0]

    print("Actual dataset ", df.shape)

    df = df.drop(index=nan_vals.keys())

    print("New dataset ", df.shape)

    print(f'Rows drop percentage {percentage}%')

    return df, percentage



def drop_columns_from_dataframe(df, col_list= ['colA','colB']):

    return df.drop(col_list, axis=1, inplace=True)



def drop_rows_based_on_indices(df, indices = None):

    return df.drop(indices, inplace = True)



def generate_condition(df,col_name='colA', value= 'valA', com_op = opt.lt): # TODO: Should be used for comparing column with a value 

    dtype_ = df[col_name].dtype

    value_ = pd.Series([value], dtype=dtype_)

    indices = df[com_op(df[col_name], value_[0])].index #['eq', 'ne', 'ge', 'le', 'gt', 'lt']

    return indices



def check_correlation_against_target_column(df, col_name, target_col_name): # TODO: Should be used for getting correlation between a target and specified column

    return df[col_name].corr(df[target_col_name])



def correaltion_dataframe(df, target_col, positive= True): # TODO: Should be used for getting correlation dataframe based on positive and negative

    correlations =  df.corr()[target_col].sort_values(ascending=False)

    if positive == True:

        return correlations[correlations > 0]

    else:

        return correlations[correlations < 0]

    

def real_discrete_columns_list(df):

    real = [i for i in range(len(df.iloc[0])) if type(df.iloc[0, i]) != str]

    discrete = [i for i in range(len(df.iloc[0])) if type(df.iloc[0, i]) == str]

    real_names = set()

    discrete_names = set()

    for i,v in enumerate(df.columns):

        if i in real:

            real_names.add(v)

        else:

            discrete_names.add(v)

    return real, discrete, real_names, discrete_names



def interpolate_missing_data(df, real_names, discrete_names):

    imputation_failed = set()

    for col in real_names:

        df[col] = df[col].fillna(df[col].median())

        if df[col].isnull().sum():

            imputation_failed.add(col)

    for col in discrete_names:

        df[col] = df[col].fillna(df[col].mode())

        if df[col].isnull().sum():

            imputation_failed.add(col)

    for col in list(imputation_failed):

        if col in discrete_names:

            val = df[col].mode().values[0]

            df[col].fillna(val,inplace=True)

        else: 

            val = df[col].median().values[0]

            df[col].fillna(val,inplace=True)

    return df



def remove_outliers(data, real_names,discrete_names):

    real_names = list(real_names)

    discrete_names = list(discrete_names)

    cat_data = data[discrete_names]

    real_data = data[real_names]

    Q1 = real_data.quantile(.25)

    Q3 = real_data.quantile(.75)

    IQR = Q3 - Q1

    new_df = real_data

    new_df_out_0 = new_df[~((new_df < (Q1 - 1.5 * IQR))|(new_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    new_df_out_1 = new_df.drop(new_df_out_0.index, inplace=False)

    percentage = 100 * new_df_out_1.shape[0] / real_data.shape[0]

    for i,col in enumerate(new_df_out_1.columns):

        new_df_out_1[col] = new_df_out_0[col].median()

    real_data = pd.concat([new_df_out_0, new_df_out_1], axis=0)

    final_data = pd.concat([real_data, cat_data], axis=1)

    percentage = 100 * final_data.shape[0] / data.shape[0]

    if final_data.shape[0] != data.shape[0]:

        print(f'Rows drop percentage {percentage}%')

    return final_data



def outlier_check(df):

    test_df = df

    Q1 = test_df.quantile(.25)

    Q3 = test_df.quantile(.75)

    IQR = Q3 - Q1

    test_df_out_1 = test_df[((test_df < (Q1 - 1.5 * IQR)) |(test_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    percentage = 100 * test_df_out_1.shape[0] / df.shape[0]

    print(f'Outlier percentage {percentage}%')

    return percentage



def label_encoding(df, discrete_names):

    le_count = 0

    le_dict = {}

    for col in discrete_names:

        if len(list(df[col].unique())) <= 3:

            col_values, unique_name = pd.factorize(df[col])

            le_dict[col] = list(unique_name)

            df[col] = col_values

            le_count += 1

    print('%d columns were label encoded.' % le_count)

    return df, le_dict



def one_hot_encoder(df, discrete_names):

    df2 = pd.get_dummies(df, columns= discrete_names)

    new_columns = set(df2.columns) - set(df.columns)

    print('%d columns were one hot encoded.' % len(new_columns))

    return df2, list(new_columns)
files = read_file_names(path='/kaggle/input')
train = load_file_into_dataframe(files['application_train.csv'])

test = load_file_into_dataframe(files['application_test.csv'])
# shape i.e [df.shape]

# info i.e [df.info()]

# checking on columnwise value frequency i.e [dict(df.count(axis=0))] 

# for the purpose of column counts, min, max, mean, Q1, Q3 we can use the above dataframe i.e [df.describe()] or [df.describe(). T] [for transposed dataframe]

# columnwise individual's datatypes (which column contains which data type) i.e [df.dtypes]

# count column number for every datatype i.e [df.dtypes.value_counts()]

# unique value for specified dataype's columns i.e [df.select_dtypes(dtype).apply(pd.Series.nunique, axis = 0)]

# unique values in specified columns i.e [df[col_name].value_counts()]

# correlation matrix i.e [df.corr()]

# check sum of null values in every column of a df i.e [df.isnull().sum()]

# check sum of not-null values in every column of a df i.e [df.notnull().sum()]

train_row, train_col =  train.shape

test_row, test_col = test.shape

print(train_row, train_col)

print(test_row, test_col)
target_column_label = 'TARGET'

identifier_column_label = 'SK_ID_CURR'



# train.drop([target_column_label], axis=1, inplace=True)

test['set'] = 'test'

test[target_column_label] = -999

train['set'] = 'train'

combined_df = train.append(test, ignore_index=True)

identifier_column_value = combined_df[identifier_column_label]

target_column_value = combined_df[target_column_label] 

combined_df.drop([identifier_column_label,target_column_label], axis=1, inplace=True)
# split dataframe between df, null df and not null df

# check missing data statistics i.e null values | percentage of total null values 

# drop columns from dataframe with column names list

# drop rows from dataframe with indices list

with timer("Splitting dataframe between dataframe, null-dataframe, not-nul-dataframe and percentage"):

    df, null_train_df, not_null_train_df, percentage = dataframe_split_between_null_and_not_null(combined_df)

print()

with timer("Splitting real data and discrete data columns"):

    real, discrete, real_names, discrete_names = real_discrete_columns_list(df)

with timer("Dropping columns based on threshold"):

    df, cols_percentage = column_dropper(df, threshold = .85)

print()

with timer("Dropping rows based on threshold"):

    df, rows_percentage = row_dropper(df, threshold = .9)

with timer("Missing value interpolation for the dataframe:"):

    df = interpolate_missing_data(df, real_names, discrete_names)

print()

with timer("Removing outliers from the dataframe:"):

    while(outlier_check(df) != 0.0):

        df = remove_outliers(df,real_names,discrete_names)

print()

with timer("Label encoding the dataframe:"):

    df, le_dict = label_encoding(df, discrete_names)

print()

with timer("One hot encoding the dataframe:"):

    df, new_columns = one_hot_encoder(df, (set(discrete_names) - set(le_dict.keys())))

print()

with timer("Assigning the label and one hot encoding variable:"):

    label_encoded_dict = le_dict

    label_encoded_columns = list(le_dict.keys())

    one_hot_encoded_columns = new_columns

print()

# train_stat_df = missing_values_stat_in_columns(application_train)

# dropped_column_train_df = drop_columns_from_dataframe(application_train, col_list= ['colA','colB'])

# dropped_rows_train_df = drop_rows_based_on_indices(application_train, indices = None)

# missing_value_manipulation(application_train,30.0)



#### Check anomalies as missing data #

#### Check outliers as missing data #

# Deleting rows and columns based on their missing values

# Replacing values with mean, median and mode

# Assigning an unique category

# Predicting the missing value

# Using algorithms which support missing values
df[identifier_column_label] = identifier_column_value

df[target_column_label] = target_column_value
# creating and entity set 'es'

es = ft.EntitySet(id = 'cr-entity')



# adding a dataframe 

es = es.entity_from_dataframe(entity_id = 'df', dataframe = df, index = 'SK_ID_CURR')
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]

default_trans_primitives =  ["day", "year", "month", "weekday", "haversine",'num_characters', 'num_words'] 
# ft.primitives.list_primitives().name.to_list()
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'df',trans_primitives = default_trans_primitives, agg_primitives=default_agg_primitives, max_depth = 2, features_only=False, verbose = True, n_jobs = 3)
# feature_matrix.columns

# feature_matrix.head()

feature_matrix = feature_matrix.reindex(index=df[identifier_column_label])

feature_matrix = feature_matrix.reset_index()

print('Saving features')

feature_matrix.to_csv('feature_matrix.csv', index = False)
# !python -m pip install featuretools