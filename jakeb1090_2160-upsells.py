# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import missingno

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from datetime import datetime

import plotly.graph_objs as go

import plotly as pl





# Any results you write to the current directory are saved as output.



# Data for plotting

#t = np.arange(0.0, 2.0, 0.01)

#s = 1 + np.sin(2 * np.pi * t)



#fig, ax = plt.subplots()

#ax.plot(t, s)





#ax.set(xlabel='time (s)', ylabel='voltage (mV)',

#       title='sin(x)')

#ax.grid()



#fig.savefig("test.png")

#plt.show()
df3.head()





df = pd.read_csv('../input/upgrade/upg.csv')

roster = pd.read_csv('../input/dashid2/inMomentnps idindex.csv')

df3 = pd.read_csv('../input/agenttab/Agenttab_data.csv')

df4 = pd.read_csv('../input/upsells/3.5-14 UPG.csv')

df.rename(columns = {'CSR ID' : 'CSR'}, inplace = True)

df.drop(columns = {'GM Area', 'Profit Ctr', 'Due Loc', 'Satellite', 'Demand Type'}, inplace = True)

roster.rename(columns = {'Nbr' : 'CSR'}, inplace = True)

results = df.merge(roster, on = 'CSR', how = 'right')

roster.drop(columns = {'Unnamed: 2', 'Unnamed: 3'}, inplace = True)



#df3.loc([['Product'] == 'Upsell'])





df[filter_bool]   



#missingno.matrix(df2, figsize = (8,3))

print(hire)

df.head()
df3['Hire_Date'] =  pd.to_datetime(df3['Hire_Date'], format='%m/%d/%Y')

df3.drop_duplicates('Agent', inplace = True)

df3 = df3.sort_values(by = 'Hire_Date', ascending = 1)



hire = df3[['Agent', 'Hire_Date']]






