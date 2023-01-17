# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python doc
# ker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Additional imports beyond the starter cell
import tensorflow as tf
import seaborn as sns
% matplotlib inline
# Import the files as dataframes
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_df = pd.read_csv('../input/sample_submission.csv')
# Check out the dataframes
df.head()
test_df.head()
sample_df.head()
# Returning to the train and test dataframes, let's take a look at our distributions
df.describe()
# Show a list of the columns
list(df.columns)
df.DateTime.value_counts()
print(df.DateTime.max())
print(df.DateTime.min())
df.shape
df.head()
# So, lots of categorical variables. How to deal with these...
# Fill in the missing name with 'No Name', because not having a name is significant.
# Not going to use the outcome subtype since it's not in the test set, so no need to fillna on that.
df['Name'].fillna('No Name', inplace=True)
df.head()
# Now, to feature engineer some of this stuff. 