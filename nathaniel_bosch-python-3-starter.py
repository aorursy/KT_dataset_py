# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np   # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.



def my_sys_call(command):

    print(check_output(command.split()).decode("utf8"))
#my_sys_call("head -2 ../input/train_categorical.csv")

#my_sys_call("head -2 ../input/train_date.csv")

#my_sys_call("head -2 ../input/train_numeric.csv")

my_sys_call("head -8 ../input/sample_submission.csv")

my_sys_call("tail -8 ../input/sample_submission.csv")
def get_headers(filename,rows=1):

    df = pd.read_csv(filename, nrows=rows)

    return df.columns.values



def get_all_headers(data_folder = "../input"):

    date_headers = get_headers(data_folder + "/train_date.csv")

    numeric_headers = get_headers(data_folder + "/train_numeric.csv")

    categorical_headers = get_headers(data_folder + "/train_categorical.csv")

    return date_headers,numeric_headers,categorical_headers
date_headers,numeric_headers,categorical_headers = get_all_headers()
def load_response_column(numeric_cols, data_folder = "../input"):

    df = pd.read_csv(data_folder + "/train_numeric.csv", 

        index_col = 0, header = 0, #nrows=100,

        usecols = numeric_cols)

    return df



numeric_cols = [0, len(numeric_headers)-1]

response_col = load_response_column(numeric_cols)
len(df)
def describe_response_column(df):

    response_levels = df['Response'].unique()

    print("Unique values in response column: {}".format(response_levels))

    X_pos = df[df['Response']==1]

    X_neg = df[df['Response']==0]

    n_rows = len(df)

    pos_rows = X_neg.shape[0]

    neg_rows = X_pos.shape[0]

    print("Rows with Response==0: {}/{}  ({}%)".format(pos_rows,n_rows,100*pos_rows/n_rows))

    print("Rows with Response==1: {}/{}  ({}%)".format(neg_rows,n_rows,100*neg_rows/n_rows))

    

describe_response_column(response_col)
submission = pd.read_csv("../input/sample_submission.csv", index_col=0)

#submission["Response"] = ...  #fill in response here

submission.to_csv("submission.csv.gz", compression="gzip")