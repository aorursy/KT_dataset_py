import numpy as np
import pandas as pd
import os
from IPython.display import display
import matplotlib.pyplot as plt

print(os.listdir("../input"))

df = pd.read_pickle('../input/bweb_1t_TO_101020182047.csv.gz.pickle', compression='gzip')
display(df.head())
ok_columns=[]
for i in df.columns:
    print("Column:", i, "Unique Values:", len(df[i].unique()),"Dtype:",df[i].dtype)
    if(len(df[i].unique())<=1): # category have bugs?
        continue
    ok_columns.append(i)
    plt.title(i + " - Unique Values: " + str(len(df[i].unique())))
    if(df[i].dtype.name != 'category'):
        plt.hist(df[i].values, bins=100)
    else:
        df[i].value_counts().plot(kind='bar')
    plt.show()
#from pandas.plotting import scatter_matrix
#scatter_matrix(df[ok_columns], alpha=0.2, figsize=(6, 6), diagonal='kde')

