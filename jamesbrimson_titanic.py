#Load the packages that we will use

import pandas as pd

import numpy as np

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
Whos
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
