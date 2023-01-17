# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pyt

import seaborn as sns



%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# read startbucks_drinkMenu_expanded.csv



df=pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')



#describe data

print(df.describe())



#print histogram of Calories

sns.distplot(df['Calories'],kde=False).set_title("Calories of Starbucks Drinks")








