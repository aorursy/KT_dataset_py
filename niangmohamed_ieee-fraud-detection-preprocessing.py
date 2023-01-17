import numpy as np # linear algebra

import pandas as pd # data processing, CSV file 



import seaborn as sns

import matplotlib.pyplot as plt





from sklearn import preprocessing

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

import matplotlib.gridspec as gridspec

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import gc

gc.enable()



import os

os.chdir('/kaggle/input/ieeecis-fraud-detection') # Set working directory

print(os.listdir('/kaggle/input/ieeecis-fraud-detection'))
%%time

train_transaction = pd.read_csv('train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('test_identity.csv', index_col='TransactionID')

print ("Data is loaded!")
print('train_transaction shape is {}'.format(train_transaction.shape))

print('test_transaction shape is {}'.format(test_transaction.shape))

print('train_identity shape is {}'.format(train_identity.shape))

print('test_identity shape is {}'.format(test_identity.shape))
%%time

train_df = pd.merge(train_transaction, train_identity, on = "TransactionID", how = "left")

print("Tain: ",train_df.shape)

del train_transaction, train_identity

gc.collect()
%%time

test_df = pd.merge(test_transaction, test_identity, on = "TransactionID", how = "left")

print("Test: ",test_df.shape)

test_df["isFraud"] = 0

del test_transaction, test_identity

gc.collect()
emails = {

'gmail': 'google', 

'att.net': 'att', 

'twc.com': 'spectrum', 

'scranton.edu': 'other', 

'optonline.net': 'other', 

'hotmail.co.uk': 'microsoft',

'comcast.net': 'other', 

'yahoo.com.mx': 'yahoo', 

'yahoo.fr': 'yahoo',

'yahoo.es': 'yahoo', 

'charter.net': 'spectrum', 

'live.com': 'microsoft', 

'aim.com': 'aol', 

'hotmail.de': 'microsoft', 

'centurylink.net': 'centurylink',

'gmail.com': 'google', 

'me.com': 'apple', 

'earthlink.net': 'other', 

'gmx.de': 'other',

'web.de': 'other', 

'cfl.rr.com': 'other', 

'hotmail.com': 'microsoft', 

'protonmail.com': 'other', 

'hotmail.fr': 'microsoft', 

'windstream.net': 'other', 

'outlook.es': 'microsoft', 

'yahoo.co.jp': 'yahoo', 

'yahoo.de': 'yahoo',

'servicios-ta.com': 'other', 

'netzero.net': 'other', 

'suddenlink.net': 'other',

'roadrunner.com': 'other', 

'sc.rr.com': 'other', 

'live.fr': 'microsoft',

'verizon.net': 'yahoo', 

'msn.com': 'microsoft', 

'q.com': 'centurylink', 

'prodigy.net.mx': 'att', 

'frontier.com': 'yahoo', 

'anonymous.com': 'other', 

'rocketmail.com': 'yahoo',

'sbcglobal.net': 'att',

'frontiernet.net': 'yahoo', 

'ymail.com': 'yahoo',

'outlook.com': 'microsoft',

'mail.com': 'other', 

'bellsouth.net': 'other',

'embarqmail.com': 'centurylink',

'cableone.net': 'other', 

'hotmail.es': 'microsoft', 

'mac.com': 'apple',

'yahoo.co.uk': 'yahoo',

'netzero.com': 'other', 

'yahoo.com': 'yahoo', 

'live.com.mx': 'microsoft',

'ptd.net': 'other',

'cox.net': 'other',

'aol.com': 'aol',

'juno.com': 'other',

'icloud.com': 'apple'

}



# number types for filtering the columns

int_types = ["int8", "int16", "int32", "int64", "float"]
# Let's check how many missing values has each column.



def check_nan(df, limit):

    '''

    Check how many values are missing in each column.

    If the number of missing values are higher than limit, we drop the column.

    '''

    

    total_rows = df.shape[0]

    total_cols = df.shape[1]

    

    total_dropped = 0

    col_to_drop = []

    

    for col in df.columns:



        null_sum = df[col].isnull().sum()

        perc_over_total = round((null_sum/total_rows), 2)

        

        if perc_over_total > limit:

            

            print("The col {} contains {} null values.\nThis represents {} of total rows."\

                  .format(col, null_sum, perc_over_total))

            

            print("Dropping column {} from the df.\n".format(col))

            

            col_to_drop.append(col)

            total_dropped += 1            

    

    df.drop(col_to_drop, axis = 1, inplace = True)

    print("We have dropped a total of {} columns.\nIt's {} of the total"\

          .format(total_dropped, round((total_dropped/total_cols), 2)))

    

    return df
def binarizer(df_train, df_test):

    '''

    Work with cat features and binarize the values.

    Works with 2 dataframes at a time and returns a tupple of both.

    '''

    cat_cols = df_train.select_dtypes(exclude=int_types).columns



    for col in cat_cols:

        

        # creating a list of unique features to binarize so we dont get and value error

        unique_train = list(df_train[col].unique())

        unique_test = list(df_test[col].unique())

        unique_values = list(set(unique_train + unique_test))

        

        enc = LabelEncoder()

        enc.fit(unique_values)

        

        df_train[col] = enc.transform((df_train[col].values).reshape(-1 ,1))

        df_test[col] = enc.transform((df_test[col].values).reshape(-1 ,1))

    

    return (df_train, df_test)
def cathegorical_imputer(df_train, df_test, strategy, fill_value):

    '''

    Replace all cathegorical features with a constant or the most frequent strategy.

    '''

    cat_cols = df_train.select_dtypes(exclude=int_types).columns

    

    for col in cat_cols:

        print("Working with column {}".format(col))

        

        # select the correct inputer

        if strategy == "constant":

            # input a fill_value of -999 to all nulls

            inputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

        elif strategy == "most_frequent":

            inputer = SimpleImputer(strategy=strategy)

        

        # replace the nulls in train and test

        df_train[col] = inputer.fit_transform(X = (df_train[col].values).reshape(-1, 1))

        df_test[col] = inputer.transform(X = (df_test[col].values).reshape(-1, 1))

        

    return (df_train, df_test)
def numerical_inputer(df_train, df_test, strategy, fill_value):

    '''

    Replace NaN in the numerical features.

    Works with 2 dataframes at a time (train & test).

    Return a tupple of both.

    '''

    

    # assert valid strategy

    message = "Please select a valid strategy (mean, median, constant (and give a fill_value) or most_frequent)"

    assert strategy in ["constant", "most_frequent", "mean", "median"], message

    

    # int_types defined earlier in the kernel

    num_cols = df_train.select_dtypes(include = int_types).columns

    

    for col in num_cols:



        print("Working with column {}".format(col))



        # select the correct inputer

        if strategy == "constant":

            inputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

        elif strategy == "most_frequent":

            inputer = SimpleImputer(strategy=strategy)

        elif strategy == "mean":

            inputer = SimpleImputer(strategy=strategy)

        elif strategy == "median":

            inputer = SimpleImputer(strategy=strategy)



        # replace the nulls in train and test

        try:

            df_train[col] = inputer.fit_transform(X = (df_train[col].values).reshape(-1, 1))

            df_test[col] = inputer.transform(X = (df_test[col].values).reshape(-1, 1))

        except:

            print("Col {} gave and error.".format(col))

            

    return (df_train, df_test)
def pipeline(df_train, df_test):

    '''

    We define a personal pipeline to process the data and fill with processing functions.

    NOTE: modifies the df in place.

    '''

    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

    # We have set the limit of 70%. If a column contains more that 70% of it's values as NaN/Missing values we will drop the column

    # Since it's very unlikely that it will help our future model.

    print("Checking for nan values\n")

    df_train = check_nan(df_train, limit=0.7)

    

    # Select the columns from df_train with less nulls and asign to test.

    df_test = df_test[list(df_train.columns)]

          

    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

          

    # mapping emails

    print("Mapping emails \n")

    df_train["EMAILP"] = df_train["P_emaildomain"].map(emails)

    df_test["EMAILP"] = df_test["P_emaildomain"].map(emails)



    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

          

    # replace nulls from the train and test df with a value of "Other"

    print("Working with cathegorical values\n")

    df_train, df_test = cathegorical_imputer(df_train, df_test, strategy = "constant", fill_value = "Other")

    

    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

          

    # now we will make a one hot encoder of these colums

    print("Binarazing values\n")

    df_train, df_test = binarizer(df_train, df_test)

    

    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

          

    # working with null values in numeric columns

    print("Working with numerical columns. NAN values\n")

    df_train, df_test = numerical_inputer(df_train, df_test, strategy = "constant", fill_value=-999)

        

    print("Shape of train is {}".format(df_train.shape))

    print("Shape of test is {}".format(df_test.shape))

          

    return (df_train, df_test)
# before preprocesing

print("Train before preprocesing: ",train_df.shape)

print("Test before preprocesing: ",test_df.shape)



train_df, test_df = pipeline(train_df, test_df)



# after preprocesing

print("Train after preprocesing: ",train_df.shape)

print("Test after preprocesing: ",test_df.shape)
# check for null values

columns = train_df.columns

for col in  columns:

    total_nulls = train_df[col].isnull().sum()

    if total_nulls > 0:

        print(col, total_nulls)

        

columns = test_df.select_dtypes(exclude=int_types).columns

train_df[columns]



columns = test_df.select_dtypes(include=int_types).columns

train_df[columns]
train_df.to_pickle('/kaggle/working/train_df.pkl')

test_df.to_pickle('/kaggle/working/test_df.pkl')