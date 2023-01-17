#Load the packages in that we will use

import pandas as pd

import numpy as np

import csv as csv

from sklearn import ensemble

from sklearn import tree

#Finding the working directory

import os

os.getcwd()
#Check the files within the working directory

from subprocess import check_output

print(check_output(["ls", "../working"]).decode("utf8"))
#Checking if files are in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Change it if not convienient 

os.chdir('/kaggle/input')

#Verify

os.getwd()
#Change it if not convienient 

os.chdir('/kaggle/input')

#Verify

os.getcwd()
train_df = pd.read_csv('train.csv', header=0)
whos
#Count number of rows and colums

train_df.shape
#Get information about variables within the dataframe

train_df.info()
#Inspect a statistical summary of dataframe

train_df.describe().transpose()

#But not all of the variables show up
#Inspect first rows

train_df.head(5)