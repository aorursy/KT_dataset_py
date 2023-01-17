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
#Load the packages that we will use

import pandas as pd

import numpy as np

import csv as csv

from sklearn import ensemble

from sklearn import tree 
import csv as csv
from sklearn import ensemble
from sklearn import tree
#Finding the working directory

import os

os.getcwd()





#Check what files are in the working directory

from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))





#Check what files are in the working directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





#Change it if not conveninent

os.chdir('/kaggle/input')





#Verify it has been changed successfully

import os

os.getcwd()





train_df = pd.read_csv('train.csv', header=0)



whos
#Count number of rows and columns

train_df.shape





#Geet information about the variables in the dataframe

train_df.info()





#Inspect a statistical summary of the dataframe

train_df.describe().transpose()

#But not all of the variables show up!





#Checking the type of variables in the dataframe

train_df.dtypes





#Inspect first rows

train_df.head(5)





#Inspect last rows

train_df.tail(5)
# female = 0, male = 1

train_df['Gender'] = train_df['Sex'].map( {'female':0, 'male':1}).astype(int)
median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age