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
df_train = pd.read_csv('../input/train_users_2.csv')

# Check variables data type
df_train.info()

 
#Create lists for categorical variables(columns) for which you want to create the dummy variables
col = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 
            'affiliate_provider','first_affiliate_tracked', 'signup_app', 'first_device_type', 
           'first_browser']

#Create a dict. object 'd' to store all dummy variable data frames
d = {name: pd.get_dummies(df_train[name], prefix=name) for name in col}

#deleting last column of each dataframe to remove multi-collinearity
for i in range(len(col)-1):
  d[col[i]].drop(d[col[i]].columns[d[col[i]].shape[1]-1], axis=1, inplace=True)

#Concat all dummy data frames
d = pd.concat(d, axis=1)

#Remove categorical variables from the training dataset
df_train.drop(col, axis=1, inplace=True)

#Concatenate all the dummy variables to the training data set and store into a new data frame
df_model = pd.concat([df_train, d], axis=1)

df_model.head() 

