# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df



X_train = import_data("../input/train.csv")

X_test = import_data("../input/test.csv")
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



X_train["seat"] = LabelEncoder().fit_transform(X_train[["seat"]])

X_train["seat"] = OneHotEncoder().fit_transform(X_train[["seat"]]).toarray()



X_train["crew"] = LabelEncoder().fit_transform(X_train[["crew"]])

X_train["crew"] = OneHotEncoder().fit_transform(X_train[["crew"]]).toarray()



X_test["seat"] = LabelEncoder().fit_transform(X_test[["seat"]])

X_test["seat"] = OneHotEncoder().fit_transform(X_test[["seat"]]).toarray()



X_test["crew"] = LabelEncoder().fit_transform(X_test[["crew"]])

X_test["crew"] = OneHotEncoder().fit_transform(X_test[["crew"]]).toarray()
y_train = X_train['event']

X_train = X_train.drop(columns=['event','experiment','time'])

X_test_id = X_test['id']

X_test = X_test.drop(columns=['id','experiment','time'])
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
import csv

predictions_file = open("sample.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["ID", "PredictedEvent"])

open_file_object.writerows(zip(X_test_id,y_pred))

predictions_file.close()