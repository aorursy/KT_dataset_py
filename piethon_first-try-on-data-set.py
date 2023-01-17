# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# create a DataFrame

df = pd.read_csv('../input/2016.csv', header=0, sep=',', index_col=False)
# see data shape

df.shape
# see data type

df.dtypes
# See data head

df.head()
df.describe()
import pandas as pd

%matplotlib inline

import random

import matplotlib.pyplot as plt

import seaborn as sns
country = df['Country']

Region = df['Region']

Happiness_Rank = df['Happiness Rank']
df_copy = df

df.drop(['Country', 'Region','Happiness Rank'], axis = 1, inplace = True)
pd.scatter_matrix(df, alpha = 0.8, figsize = (15,15), diagonal = 'kde');
pd.scatter_matrix(df, alpha = 0.8, figsize = (15,15), diagonal = 'kde');
df.drop(['Freedom', 'Trust (Government Corruption)','Generosity','Dystopia Residual'], axis = 1, inplace = True)
pd.scatter_matrix(df, alpha = 0.8, figsize = (12,12), diagonal = 'kde');