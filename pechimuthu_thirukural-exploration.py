# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#change current directory to dataset directory
os.chdir("../input")

# Any results you write to the current directory are saved as output.
df=pd.read_csv('Thirukural.csv')

df_exp=pd.read_csv('Thirukural With Explanation.csv')
#explore
df.head()
#replacing tabs with spaces to read clearly
df['Verse']=df['Verse'].str.replace('\t',' ')
df.head()
#section name is Athigaram

#Capture how many sections in each chapter
df['Chapter Name'].value_counts()
df_exp.head(2)
# I dont see more difference between df and df_exp than an Explanation column.

#Adding the Explanation column to df.
df.loc[:,'Explanation']=df_exp.loc[:,'Explanation']
df.head(2)
