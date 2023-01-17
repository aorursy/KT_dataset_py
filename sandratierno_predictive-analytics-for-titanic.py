# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Load the packages that we will use



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import csv as csv





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn import ensemble # random forest 

from sklearn import tree # tree 

# Any results you write to the current directory are saved as output.
#Finding the working directory



import os

os.getcwd()
#Check what files are in the working directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#change if it is not convenient

os.chdir('/kaggle/input')



#Verify it has been changed succesfully

os.getcwd()
train_df = pd.read_csv('train.csv', header=0)
whos
#Count number of rows and columns

train_df.shape
train_df.info()
train_df.describe
train_df.describe().transpose()
train_df.head(5)
