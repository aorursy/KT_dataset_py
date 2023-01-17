import os

import pandas as pd



#Garbage Collector

import gc
#Read in the data

train_id = pd.read_csv('../input/train_identity.csv')

train_tran = pd.read_csv('../input/train_transaction.csv')
#This is a function that downcast the integer columns

def downcast_df_int_columns(df):

    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)

        

    if len(list_of_columns)>=1:

        max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing

        print("downcasting integers for:", list_of_columns, "\n")

        

        for col in list_of_columns:

            print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],

                  "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")

            df[col] = pd.to_numeric(df[col], downcast="integer")

            print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))

    else:

        print("no columns to downcast")

    

    gc.collect()

    

    print("done")
#This is a function that downcast the float columns, if you have too many columns to adjust and do not want to see to many messages proceesing, you could comment our the print() columns

def downcast_df_float_columns(df):

    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)

        

    if len(list_of_columns)>=1:

        max_string_length = max([len(col) for col in list_of_columns]) # finds max string length for better status printing

        print("downcasting float for:", list_of_columns, "\n")

        

        for col in list_of_columns:

            print("reduced memory usage for:  ", col.ljust(max_string_length+2)[:max_string_length+2],

                  "from", str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8), "to", end=" ")

            df[col] = pd.to_numeric(df[col], downcast="float")

            print(str(round(df[col].memory_usage(deep=True)*1e-6,2)).rjust(8))

    else:

        print("no columns to downcast")

    

    gc.collect()

    

    print("done")
train_id.info(memory_usage="deep")
train_tran.info(memory_usage="deep")
#Reducing Size for the numeric columns for the train id part

downcast_df_int_columns(train_id)

downcast_df_float_columns(train_id)
#Reducing Size for the numeric columns for the train transaction part

downcast_df_int_columns(train_tran)

downcast_df_float_columns(train_tran)
train_id.info(memory_usage="deep")
train_tran.info(memory_usage="deep")
pd.set_option('display.max_columns', None)



train_id.head(10)
#This is a function that convert a list of columns from object to categorical

def convert_columns_to_catg(df, column_list):

    for col in column_list:

        print("converting", col.ljust(30), "size: ", round(df[col].memory_usage(deep=True)*1e-6,2), end="\t")

        df[col] = df[col].astype("category")

        print("->\t", round(df[col].memory_usage(deep=True)*1e-6,2))
#Picking some columns that seems to be able to convert to categorical

convert_columns_to_catg(train_id, ['id_12', 'id_16', 'id_30', 'id_31', 'id_33', 'id_34', 'id_34', 

                                  'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo'])
train_id.info()
train_id.to_pickle("train_id.pkl")



print("train_identity.csv:", os.stat('../input/train_identity.csv').st_size * 1e-6)

print("train_id.pkl:", os.stat('train_id.pkl').st_size * 1e-6)