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


df = pd.read_csv('../input/datasets.csv', index_col=0)



%matplotlib inline

import matplotlib.pyplot as plt



df.shape
df
dfColumns = pd.read_csv('../input/datasets_columns.csv', index_col=0)

dfBounds = dfColumns.loc['lists']

dfBounds = dfBounds.set_index('bounds')

ax = dfBounds.plot.line(

    figsize=(12,5),

    fontsize=16

)

ax.set_title("Number of lists varying the trie bound", fontsize=20)
dfBounds = dfColumns.loc['avr_size']

dfBounds = dfBounds.set_index('bounds')

ax = dfBounds.plot.line(

    figsize=(12,5),

    fontsize=16

)

ax.set_title("Average of words per lists varying the trie bound", fontsize=20)