# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt


# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/countries of the world.csv")
df.head(10)
df.shape
df.describe()
df.info()
df.info()
columns = df.columns[2:]
for col in columns:
    if (df[col].dtype == "object"):
        df[col] = df[col].str.replace(",",".")
        df[col] = pd.to_numeric(df[col])
df.info()
df.describe()
correlations = df.corr()
correlations
related_columns_set = []
pair_columns = []
for col in correlations.columns:
    #print(col)
    for idx in correlations.index:
        #print(idx)
        if(correlations.loc[idx,col] >  0.4 or correlations.loc[idx,col] < -0.4) and correlations.loc[idx,col] != 1:
            if(idx+"-"+col+":"+str(correlations.loc[idx,col]) not in related_columns_set):
                related_columns_set.append(col+"-"+idx+":"+str(correlations.loc[idx,col]))
                pair_columns.append([col,idx])
                print(col,"-", idx,":", correlations.loc[idx,col])

print(len(pair_columns))
pair_columns
fig, axes = plt.subplots(nrows=14, ncols=2, figsize=(16,48))

i = 0
for pair in pair_columns:    
    #print(i//2, i%2)
    df.plot(x=pair[0],y=pair[1], kind="scatter", ax=axes[i//2, i%2])
    i += 1
