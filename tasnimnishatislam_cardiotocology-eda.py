!pip install pycaret
#import

import pandas as pd

import missingno

from pycaret.classification import *

import numpy as np
df = pd.read_csv("../input/fetalhr/CTG.csv")

df.head()
#plot graphic of missing value

missingno.matrix(df, figsize = (30, 10))
df.isnull().sum()

#there are some actually
my_data = setup(data = df, 

                numeric_imputation = "median", 

                categorical_imputation = "mode", 

                target = "NSP", train_size = 0.8, 

                transformation = True, 

                transformation_method = "yeo-johnson",

               )
#Trying to Understand my_data

print(type(my_data))

print(np.shape(my_data))

print(my_data[1])

#I don't understand much, help me in the comments