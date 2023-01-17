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
print("loading data takes about 1 minute....")



train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

#test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



#train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

#test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')



#sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



print("loading successful!")
print("shape of train_transaction: ", train_transaction.shape, "\n")



print("info of train_transaction: \n")



print(train_transaction.info())
# lets generate some useful lists of columns,

# we want a list of numerical features

# and a list of categorical features



c = (train_transaction.dtypes == 'object')

n = (train_transaction.dtypes != 'object')

cat_cols = list(c[c].index)

num_cols = list(n[n].index) 



print(cat_cols, "\n")

print("number categorical features: ", len(cat_cols), "\n\n")

print(num_cols, "\n")

print("number numerical features: ", len(num_cols))
# the int/float datatypes have the following ranges:



#   int8:  -128 to 127, range = 255  



#  int16:  -32,768 to 32,767, range = 65,535



#  int32:  -2,147,483,648 to 2,147,483,647, range = 4,294,967,295



#  int64:  -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807,

#           range = 18,446,744,073,709,551,615





#  These ranges are the same for all float datatypes.

#  By default all numerical columns in pandas are in int64 or float64.

#  This means that when we find a numerical integer column whose 

#  values do not exceed one of the ranges shown above, we can then

#  convert this datatype down to a smaller one. 
print("train_transaction.info(): \n")



print(train_transaction.info())
#  this function detects all the numerical columns,

#  that can be converted to a smaller datatype.



def detect_num_cols_to_shrink(list_of_num_cols, dataframe):

 

    convert_to_int8 = []

    convert_to_int16 = []

    convert_to_int32 = []

    

    #  sadly the datatype float8 does not exist

    convert_to_float16 = []

    convert_to_float32 = []

    

    for col in list_of_num_cols:

        

        if dataframe[col].dtype in ['int', 'int8', 'int32', 'int64']:

            describe_object = dataframe[col].describe()

            minimum = describe_object[3]

            maximum = describe_object[7]

            diff = abs(maximum - minimum)



            if diff < 255:

                convert_to_int8.append(col)

            elif diff < 65535:

                convert_to_int16.append(col)

            elif diff < 4294967295:

                convert_to_int32.append(col)   

                

        elif dataframe[col].dtype in ['float', 'float16', 'float32', 'float64']:

            describe_object = dataframe[col].describe()

            minimum = describe_object[3]

            maximum = describe_object[7]

            diff = abs(maximum - minimum)



            if diff < 65535:

                convert_to_float16.append(col)

            elif diff < 4294967295:

                convert_to_float32.append(col) 

        

    list_of_lists = []

    list_of_lists.append(convert_to_int8)

    list_of_lists.append(convert_to_int16)

    list_of_lists.append(convert_to_int32)

    list_of_lists.append(convert_to_float16)

    list_of_lists.append(convert_to_float32)

    

    return list_of_lists
num_cols_to_shrink_trans = detect_num_cols_to_shrink(num_cols, train_transaction)



convert_to_int8 = num_cols_to_shrink_trans[0]

convert_to_int16 = num_cols_to_shrink_trans[1]

convert_to_int32 = num_cols_to_shrink_trans[2]



convert_to_float16 = num_cols_to_shrink_trans[3]

convert_to_float32 = num_cols_to_shrink_trans[4]



print("convert_to_int8 :", convert_to_int8, "\n")

print("convert_to_int16 :", convert_to_int16, "\n")

print("convert_to_int32 :", convert_to_int32, "\n")



print("convert_to_float16 :", convert_to_float16, "\n")

print("convert_to_float32 :", convert_to_float32, "\n")
print("starting with converting process....")



# convert the datatypes with .astype() 



for col in convert_to_int16:

    train_transaction[col] = train_transaction[col].astype('int16')  

    

for col in convert_to_int32:

    train_transaction[col] = train_transaction[col].astype('int32') 



for col in convert_to_float16:

    train_transaction[col] = train_transaction[col].astype('float16')

    

for col in convert_to_float32:

    train_transaction[col] = train_transaction[col].astype('float32')

    

print("successfully converted!")
print("train_transaction.info(): \n")   # now uses 548 MB



print(train_transaction.info(), "\n")
for i in cat_cols:

    

    train_transaction[i] = train_transaction[i].astype('category')

    

print("successfully converted all categorical features!")
print("train_transaction.info() \n")



print(train_transaction.info(), "\n")