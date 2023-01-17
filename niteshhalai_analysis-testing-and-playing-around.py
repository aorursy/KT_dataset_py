# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/BlackFriday.csv')

df.head(10)
df.info()
df.describe()
df['Purchase'].describe()
df['Purchase'].hist(bins=50)
df.boxplot(column='Purchase')
df[df['Purchase']>20000]['Purchase'].hist(bins=50)

df[df['Purchase']>22000]['Purchase'].hist(bins=50)
df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Age')
df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Gender')
df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Product_ID')
df[df['Purchase']>22000].boxplot(column = 'Purchase', by = 'Product_Category_1')
df[df['Purchase']<22000].boxplot(column = 'Purchase', by = 'Product_Category_1')
df[df['Product_Category_1']==10]['Purchase'].hist(bins=50)
df[df['Product_Category_1']==10].boxplot(column = 'Purchase', by = 'Product_ID')