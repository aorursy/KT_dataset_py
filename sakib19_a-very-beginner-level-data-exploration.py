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
import matplotlib.pyplot as plt
filename="../input/bd-food-rating.csv"
df=pd.read_csv(filename)
df.head(20)
df.dtypes
for column in df.columns.values.tolist():

    print(column)

    print (df[column].value_counts())

    print("")    
rating = {'A+': 2,'A': 1,'B':0}

df.bfsa_approve_status = [rating[item] for item in df.bfsa_approve_status] 

df



df.loc[df['bfsa_approve_status'] == 2]

df.loc[df['bfsa_approve_status'] == 1]
df.loc[df['bfsa_approve_status'] == 0]