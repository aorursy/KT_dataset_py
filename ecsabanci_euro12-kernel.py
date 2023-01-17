# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/euro12/Euro_2012_stats_TEAM.csv")

data.head()
data.tail()
data.info()
data.dtypes
data.head()
#first of all we need to convert Shooting Accuracy value into float data type.

convert_SA = lambda x : float(x[:-1])



data["Shooting Accuracy"] = data["Shooting Accuracy"].apply(convert_SA)
data.head()
#also we need to convert % Goals-to-shots and Saves-to-shots ratio values into float data type.

convert_SA = lambda x : float(x[:-1])

data["% Goals-to-shots"] = data["% Goals-to-shots"].apply(convert_SA)

data["Saves-to-shots ratio"] = data["Saves-to-shots ratio"].apply(convert_SA)
data.head()
# now we can use describe method for the all values that we have

data.describe()
data.head()
#we are comparing Shots on target that are scored as goal or not

data.boxplot(column='Shots on target',by = 'Goals')
data_new = data.head()    

data_new
# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Team', value_vars= ['Shots on target','Shots off target'])

melted
# lets conc 2 data frame in row

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 

conc_data_row
#lets conc two data frame in column



# Firstly lets create 2 data frame

data1 = data.Team

data2= data["Players Used"]

conc_data_cols = pd.concat([data1,data2],axis =1) 

conc_data_cols.head()
data.head()
data.plot(kind='scatter', x='Shots on target', y='Shots off target',alpha = 0.5,color = 'red')

plt.xlabel('Shots on target')             

plt.ylabel('Shots off target')

plt.title('Shots on target vs Shots off target')
data["Shots on target"].plot(kind = 'line',alpha = 0.8,color = 'red',linewidth=1,grid=True,linestyle=":")

data["Shots off target"].plot(kind = 'line',alpha = 0.8,color = 'blue',linewidth=1,grid=True,linestyle="-.")



plt.xlabel('Shots on target')              

plt.ylabel('Shots off target')

plt.title('Shots on target vs Shots off target')