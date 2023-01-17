# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(os.listdir('/kaggle/input/osmi-mental-health-in-tech-survey-2019/'))
DATA_PATH = r'/kaggle/input/osmi-mental-health-in-tech-survey-2019/OSMI 2019 Mental Health in Tech Survey Results - OSMI Mental Health in Tech Survey 2019.csv'
df = pd.read_csv(DATA_PATH)
df.head()
def check_missing(column, check_df):

    if(check_df[column].isnull().any()):

        print("Number of rows having the NULL value in ",check_df," are ",check_df[column].isnull().sum())

    else:

        print("There is no NULL vale in the column", column)
def check_missing_all(check_df, verbose=False):

    empty_list = []

    for column in df.columns:

        if(check_df[column].isnull().any()):

            empty_rows = check_df[column].isnull().sum()

            empty_list.append(empty_rows)

            

            if(verbose == True):

                print("Number of rows having the NULL value in column %s are %d " %(column, empty_rows))

        else:

            print("There is no NULL value in the column %s" %(column))

            empty_list.append(0)

            

    return empty_list
def convert_columns(df):

    ''' Given a datarame numbers the columns'''

    column_names = {}

    start_i = 1

    i = 1

    for column_name in df.columns:

        column_names[column_name] = str(i)

        i += 1

    

    df.rename(column_names, axis=1, inplace=True)

    print("Columns renamed strating with i = %d to i = %d" %(start_i,i))

    return df, column_names

#         print(column_name)

    
empty_rows = check_missing_all(df, verbose=True)
print(len(df))

print(df.shape)

print(len(empty_rows))

print(empty_rows)
df, names = convert_columns(df)
def clean_missing(df, column_list, empty_rows, percent_del=0.65):

    count = 0

    dropped_columns = []

    for i, column in enumerate(column_list):

        calc_percent = empty_rows[i] / len(df)

        if(calc_percent > percent_del):

            df.drop([column], axis=1, inplace=True)

            dropped_columns.append(column)

            print("%s Column dropped "%(column))

            count += 1

    

    print("Number of columns dropped %d"%(count))



    return df, dropped_columns
print(list(df.columns))
df, dropped_columns = clean_missing(df, list(df.columns), empty_rows, percent_del=0.65)
print(list(df.columns))
def encode_categorical(df, column_list):

    for column in column_list:

        df[column] = df[column].astype('str')

        encoder = preprocessing.LabelEncoder()

        encoded_list = encoder.fit_transform(df[column])

#         print(encoded_list)

#         print(len(encoded_list))

        encoded_series = pd.Series(encoded_list)

        df[column] = encoded_series

        print("The ", column, "is encoded ")

    return(df)
# Numeric types need to do MinxMaxScaler

def scale_data(df, column_list, index_list):

    for column in column_list:

        df[column] = df[column].astype('float')

        encoder = preprocessing.StandardScaler()

        df[column] = encoder.fit_transform(df[column].values.reshape(-1,1))

        print("The ",column, "is encoded")

    return(df)
df.head()
# print(names)

print(df.columns)
pp.ProfileReport(df)