import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



def load(N=None):

    return pd.read_csv('../input/train_date.csv', engine='c', nrows=N, index_col='Id')



# load some data ( you can increase - pattern stays the same)



df = load(100000)

df.shape
# conver to boolean and drop similar rows and adjacent columns

df = np.logical_not(df.isnull())



df = df.drop_duplicates()

df = df.T.drop_duplicates().T

df.shape



# pretty less
# visualize
def map(df):

    fig, ax = plt.subplots()

    heatmap = ax.pcolor(df, cmap=plt.cm.Blues)



    # want a more natural, table-like display

    ax.invert_yaxis()

    ax.xaxis.tick_top()



    ax.set_xticklabels(df.columns, minor=False)

    ax.set_yticklabels(df.index, minor=False)

    return plt
map(df).show()



# there is a pattern - obviously
# let's sort



df = df.sort_values(list(df.columns))

map(df).show()



# wow!
# let's sort the tails



df = df.sort_values(list(df.columns[-20:-1]))

map(df).show()