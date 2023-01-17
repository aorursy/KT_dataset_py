# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# This is a set of helper functions to help you fetch data and save it in usable files



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn 

import matplotlib

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def load_user_logs():

    print("Loading training Data")

    train = pd.read_csv("../input/train.csv")

    print("Loading test Data")

    test = pd.read_csv("../input/sample_submission_zero.csv")

    valid_msno = pd.concat([train, test])['msno'].as_matrix()

    train = []

    test = []



    user_logs_iter = pd.read_csv("../input/user_logs.csv", chunksize=10000000, iterator=True, low_memory=False, parse_dates=['date'])

    user_logs = pd.DataFrame()

    user_log_counts = pd.DataFrame()



    i = 0

    for df in user_logs_iter:

        df = df[df['msno'].isin(valid_msno)]

        if i is 0: 

            user_logs = df

            user_log_counts = pd.DataFrame(user_logs['msno'].value_counts().reset_index())

            user_log_counts.columns = ['msno','logs_count']

        else:

            temp_user_log_counts = pd.DataFrame(df['msno'].value_counts().reset_index())

            temp_user_log_counts.columns = ['msno','logs_count']

            user_logs = pd.concat([user_logs, df])

            user_log_counts = pd.concat([user_log_counts, temp_user_log_counts])

        

        user_logs = user_logs.groupby(["msno"], as_index=False)["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"].sum()        

        user_log_counts = user_log_counts.groupby(["msno"], as_index=False)["logs_count"].sum()        



        print("User Logs {} df: {}".format(str(i), user_logs.shape))

        print("User Log Counts {} df: {}".format(str(i), user_log_counts.shape))     

        i = i+1

            

    print(user_logs.shape)

    user_logs = pd.merge(user_logs, user_log_counts, how='left', on='msno')

    print(user_logs.shape)

    

    return user_logs



## Suggest commenting out all logs except for the user_logs when fetching user logs

## and vice versa

def raw_daq():

#     print("Loading training Data")

#     train = pd.read_csv("../input/train.csv")

#     print("Loading test Data")

#     test = pd.read_csv("../input/sample_submission_zero.csv")

#     print("Loading transaction Data")

#     transactions = pd.read_csv("../input/transactions.csv")

#     print("Loading members Data")

#     members = pd.read_csv("../input/members.csv")

    print("Loading user Log Data")

    user_logs = load_user_logs()



#     print("Train set shape {}".format(train.shape))

#     print("Test set shape {}".format(test.shape))

#     print("Transactions set shape {}".format(transactions.shape))

#     print("Members set shape {}".format(members.shape))

    print("User Logs set shape {}".format(user_logs.shape))

    user_logs.to_csv("user_logs_collapsed.csv",header=True)
def check_sets():

    print("Test set shape {} and train set shape {}".format(test_set.shape,train_set.shape))



# Function for outputting data to csv files: Use as needed

def raw_data_merge():

    print("Merging Transactions")

    transaction_counts = pd.DataFrame(transactions['msno'].value_counts().reset_index())

    transaction_counts.columns = ['msno','trans_count']

    # Merging based on msno keys on the left input dataframe

    train_set = pd.merge(train, transaction_counts, how='left', on='msno') 

    test_set = pd.merge(test, transaction_counts, how='left', on='msno')



    print("Merging Member Data")

    gender = {'male':1, 'female':2}

    members['gender'] = members['gender'].map(gender)

    members.fillna(0)

    train_set = pd.merge(train_set, members, how='left', on='msno')

    test_set = pd.merge(test_set, members, how='left', on='msno')



    # print("Merging User Logs")

    # train_set = pd.merge(train_set, user_logs, how='left', on='msno')

    # test_set = pd.merge(test_set, user_logs, how='left', on='msno')

    train_set.to_csv("train_set_no_logs.csv",header=True)

    test_set.to_csv("test_set_no_logs.csv",header=True)

    check_sets()
print("Getting raw data")

raw_daq()