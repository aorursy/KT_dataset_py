#import the basic packages which is necessary for every model.



import numpy as np

import pandas as pd 

#On top right side select the filename of our data.

#Here my file name is "textfile". Copy the path of that file and follow the below for reading the dataset.



myfile= pd.read_csv('../input/textdata.txt') # load data from csv or txt, this processn is applicable for both train and test datasets



print(myfile) # Shows the data  
#shows the length of the dataset.

myfile.shape 
 #explains about the dataset unique words, count of words etc.

myfile.describe()