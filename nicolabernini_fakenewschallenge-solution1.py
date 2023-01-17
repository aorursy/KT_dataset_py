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
nRowsRead = 1000 # specify 'None' if want to read whole file

# data.csv has 4009 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
trusted_sites = ["www.bbc.com", "www.reuters.com", "www.nytimes.com", "www.abcnews.com", "cnn.com", "abcnews.go.com", "bbc.co.uk"]
def is_trusted(x): 

    for t in trusted_sites: 

        if t in x: 

            return 1 

    return 0 

        
df1["is_trusted_url"]=df1["URLs"].apply(is_trusted)
df1.head()
df1["Label"].corr(df1["is_trusted_url"])
def predict(x): 

    return is_trusted(x["URLs"])

n_test_set = 100

test_set = df1.sample(n=n_test_set)



# Let's pass all the possible features 

test_set_x = test_set[["URLs", "Headline", "Body"]]



# Label

test_set_y = test_set[["Label"]]
test_set_x.head()
test_set_y.head()
# This passes to the predict function the columns (so axis=1) hence iterates over the rows (axis=0)

test_set_y_pred = test_set_x.apply(predict, axis=1)
test_set_y_pred.head()
res = test_set_y_pred.values.reshape(-1,1) - test_set_y.values.reshape(-1,1)
# Any FP 

fp=res[res==+1].shape[0]

fn=res[res==-1].shape[0]

print(f"Num FP={fp}, Num FN={fn}")