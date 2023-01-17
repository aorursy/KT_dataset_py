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

import matplotlib.cm as cm

%matplotlib inline



import seaborn as sns

sns.set_style("dark")
# Method to set size for figure

def set_fig_size(w=15,l=10):

  plt.figure(figsize=(w,l))
!ls
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df.head(10)
df.isnull().sum(axis = 0)
df['type'].value_counts()
df.groupby(['type']).sum().reset_index()
set_fig_size(10,5)

sns.barplot(x='type',y='isFraud',data=df.groupby(['type']).sum().reset_index())
dfFlagged = df.loc[df.isFlaggedFraud == 1]
print("Minimum amount flagged by isFlagged is: ", dfFlagged['amount'].min())