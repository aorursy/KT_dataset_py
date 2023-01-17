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
# Environment set up
# import sys
# !{sys.executable} -m pip install pandas_profiling
# This kernal uses the following libraries
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns', 200) # Good for wide datasets - otherwise it will truncate the data in views like head
import pandas_profiling # for exploration of datasets

# Setting the seed for reproducibility
np.random.seed(42)
import os
# The kaggle challenge provides the following data:
print(os.listdir("../input"))
# Read in train and test data
train = pd.read_csv('../input/TrainData.csv',low_memory=False)
test = pd.read_csv('../input/TestData.csv',low_memory=False)
# How many samples and features do we have in the training data?
print(train.shape, test.shape)
# How should we format our predictions for submissions?
example = pd.read_csv('../input/SubmissionFormat.csv',low_memory=False)
example.head()
# Note that log is a log(x+1) transformation
# what are the features in the training data
train.info()
# Quick summary statistics for numeric features
train.describe().T
# Note that there are NEGATIVE values in the target column in the training dataset. 
# These will need to be transformed prior to using log(x+1) (e.g. setting negative values to zero)
train[train['donations_and_bequests']<0].shape
# For a quick overview of all the features, levels etc, along with the numeric correlations, 
# I'll use pandas profiler. You can check out this massive time-saver here: 
# https://github.com/pandas-profiling/pandas-profiling
profile = pandas_profiling.ProfileReport(train)
profile.to_file(outputfile="charity_data_profile.html") # download it for viewing in the browser
profile # ATTEMPT to render it in kernal