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
Train = pd.read_csv('../input/train_V2.csv') # Read the training data file from kaggle dataset

Test = pd.read_csv('../input/test_V2.csv') # Read the testing dataset file form kaggle dataset



# Change the file path to any local system to use it locally
Train # check training file 
Test # check testing file 
dfTrain = pd.DataFrame(Train) # make a dataframe using pandas of training dataset

dfTest = pd.DataFrame(Test) # make a datafarame using pandas of testing dataset
trRows = 100000 # Total instances 4446966 (training set)

for i in range(1,6):

    tr = dfTrain.iloc[(i-1)*trRows:i*trRows] # Select the specified instances in a particular iteration.

    path = "train_100k_"+str(i)+".csv" # Change the file name in every iteration,File path can also be given with name.

    tr.to_csv(path,index=False) # Convert it into CSV file.

    print(tr) # Just to check whether it's working or not.
teRows = 40000 # Total instances 1934174 (testing set)

for i in range(1,6):

    te = dfTest.iloc[(i-1)*teRows:i*teRows] # Select the specified instances in a particular iteration.

    path = "test_40K_"+str(i)+".csv" # Change the file name in every iteration,File path can also be given with name.

    te.to_csv(path,index=False) # Convert it into CSV file.

    print(te) # Just to check whether it's working or not.