# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from time import time

from sklearn.metrics import f1_score



# Read data

voice_data = pd.read_csv("../input/voice.csv")

print ("Voice data read successfully!")
# TODO: Calculate number of data points

n_datapoints = voice_data.shape[0]



# TODO: Calculate number of features

n_features = voice_data.shape[1]-1





# Print the results

print ("Total number of datapoints: ")

print (n_datapoints)

print ("Number of features: ")

print (n_features)

# Extract feature columns

feature_cols = list(voice_data.columns[:-1])



# Extract target column 'passed'

target_col = voice_data.columns[-1] 



# Show the list of columns

print ("Feature columns:\n")

print (feature_cols)

print ("\nTarget column: ")

print (target_col)



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = voice_data[feature_cols]

y_all = voice_data[target_col]



# Show the feature information by printing the first five rows

print ("\nFeature values:")

print (X_all.head())
# Import any additional functionality you may need here



from sklearn.cross_validation import train_test_split



# Set the number of training points

num_train = 2500



# Set the number of testing points

num_test = X_all.shape[0] - num_train



# TODO: Shuffle and split the dataset into the number of training and testing points above

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=float(num_test)/X_all.shape[0], random_state=42)





# Show the results of the split

print ("Training set has samples ")

print (X_train.shape[0])

print ("Testing set has samples ")

print (X_test.shape[0])