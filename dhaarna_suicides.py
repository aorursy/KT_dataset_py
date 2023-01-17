# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Suicides_in_India.csv")
df=df[df.Age_group!='0-100+']
df
df[['Age_group', 'Total']].groupby(['Age_group']).sum().plot(kind='bar')
df[['Gender', 'Total']].groupby(['Gender']).sum().plot(kind='bar', color='r')
df[['State', 'Total']].groupby(['State']).sum().plot(kind='bar', color='g')
df[['Year', 'Total']].groupby(['Year']).sum().plot(kind='bar', color='y')
df[df.Type_code=='Causes'][['Type', 'Total']].groupby(['Type']).sum().plot(kind='bar')
df[df.Type_code=='Means_adopted'][['Type', 'Total']].groupby(['Type']).sum().plot(kind='bar')
mdf=df[df.State=='Maharashtra']
mdf[mdf.Type_code=='Causes'][['Type', 'Total']].groupby(['Type']).sum().plot(kind='bar')