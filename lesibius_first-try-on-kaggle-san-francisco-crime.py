#Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Get data

df = pd.read_csv('../input/train.csv',header=0)

df_test = pd.read_csv('../input/test.csv',header=0)

df.info()
### Output
print("Number of crime categories: {}".format(len(df.Category.drop_duplicates())+1))

print(df.Category.drop_duplicates())