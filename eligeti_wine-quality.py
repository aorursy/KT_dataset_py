# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# helper functions

def hprint(string):
    """used to print headings"""
    print(str(string) + ':\n' + '=' * (len(str(string)) + 1))

def num_unique(df,c_list):
    """prints number of unique values for each column"""
    
    width_1 = 40
    width_2 = 10
    colan_sym=':'
    hprint('Number of Unique values')
    for c in c_list:
        col = df[str(c)]
        print(f'{c} {colan_sym:>{width_1 - len(c)}}  {col.nunique():>{width_2}}')
        
def unique_val(df,c_list):
    hprint('List of unique values in columns')
    for _ in c_list:
        print(f'{_}:{df[_].unique()}')
     
    
def c_proportions(df, c_list):
    width_1 = 30
    width_2 = 5
    
    
    precision_2 = 3
    
    for c in c_list:
        col=df[str(c)]
        a = (col.value_counts())
        string_1 = '---'+str(c)+'---'
        print(string_1)
        for i in range(len(a.index)):
            string_1 = a.index[i]
            value_2 = (a.values[i]/a.sum()) *100
            print(f"{string_1:>{width_1}}: {value_2:{width_2}.{precision_2}}%")
        print("\n"*3)
hprint("Current directory")
os.listdir()
hprint("Parent directory")
os.listdir("../")
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
hprint("head")
df.head()
hprint('Shape')
df.shape
hprint('Number of NA values in each column')
df.isna().sum()
hprint('dtypes')
df.dtypes
num_unique(df,df.columns)
unique_val(df, ['quality'])
df['quality'].value_counts()
df.describe()
df.corr()
InteractiveShell.ast_node_interactivity = "last"
fig, ax = plt.subplots(figsize=(8,8))
mask = np.triu(df.corr())
ax = sns.heatmap(df.corr(),cmap='RdBu',cbar=True, square=True,linecolor='white', linewidth=0.1,mask=mask)
# Hypothesis Testing
# Quality & volatile acidity

