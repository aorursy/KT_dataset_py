# Compatibility layer between Python 2 and Python 3

from __future__ import print_function

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn import preprocessing



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D

from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

from keras.utils import np_utils



# %%
def convert_to_float(x):



    try:

        return np.float(x)

    except:

        return np.nan



def read_data(file_path):



    """

    This function reads the accelerometer data from a file

    Args:

        file_path: URL pointing to the CSV file

    Returns:

        A pandas dataframe

    """



    column_names = ['user-id',

                    'activity',

                    'timestamp',

                    'x-axis',

                    'y-axis',

                    'z-axis']

    df = pd.read_csv(file_path,

                     header=None,

                     names=column_names)

    # Last column has a ";" character which must be removed ...

    df['z-axis'].replace(regex=True,

      inplace=True,

      to_replace=r';',

      value=r'')

    # ... and then this column must be transformed to float explicitly

    df['z-axis'] = df['z-axis'].apply(convert_to_float)

    # This is very important otherwise the model will not fit and loss

    # will show up as NAN

    df.dropna(axis=0, how='any', inplace=True)



    return df



def feature_normalize(dataset):



    mu = np.mean(dataset, axis=0)

    sigma = np.std(dataset, axis=0)

    return (dataset - mu)/sigma
#Load the data

df = read_data('../input/wisdm-ar-v11-raw/WISDM_ar_v1.1_raw.txt')

#Display the shape of data

#print(df)

#Display the number of columns of data

print("Number of columns in the dataframe: %i" % (df.shape[1]))

#Display the number of rows of data

print("Number of rows in the dataframe: %i\n" % (df.shape[0]))

#Display the first 20 rows of data

print("First 20 rows of the dataframe:\n")

print(df.head(50))
# Define column name of the label vector

LABEL = "ActivityEncoded"

# Transform the labels from String to Integer via LabelEncoder

le = preprocessing.LabelEncoder()

# Add a new column to the existing DataFrame with the encoded values

df[LABEL] = le.fit_transform(df["activity"].values.ravel())

df
# Differentiate between test set and training set

df_test = df[df['user-id'] > 28]

df_train = df[df['user-id'] <= 28]
# Normalize features for training data set

df_train['x-axis'] = feature_normalize(df['x-axis'])

df_train['y-axis'] = feature_normalize(df['y-axis'])

df_train['z-axis'] = feature_normalize(df['z-axis'])

# Round in order to comply to NSNumber from iOS

df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

df_train