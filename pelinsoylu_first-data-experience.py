# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#----- These are the default lines------

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.describe()
data.corr()
#correlation 

#import seaborn lib. for heatmap.

fig,ax = plt.subplots(figsize=(20,20)) # adjust your figure size 

sns.heatmap(data.corr(), annot=True, linewidths=.10, fmt= '.1f',ax=ax,cmap="YlGnBu") #If annot=True, write the data value in each cell."vmin=0, vmax=1" Change the limits of the colormap

plt.show()
data.head(3) #first 3 data.If you don't use parameter,it shows 5 of them. 
data.tail() #last 5 data
# Scatter Plot 

data.plot(kind='scatter',x='V1',y='Amount',color='red',alpha=0.3) #alpha: transperancy

plt.xlabel='V1'

plt.ylabel='Amount'

plt.show()



# Histogram

# bins = number of bar in figure

data.plot(kind = 'hist',bins = 30,figsize = (12,12))

plt.show()

#plt.clf()----->clf() = cleans it.
series= data['Amount']

print(type(series))

data_frame= data[['Amount']]

print(type(data_frame))
x=data['V1']>2.4

data[x]
#filtering

data[(data.V1>2.0) & (data.V2>1.3)]

filter1= data.Amount>1.9

filter2=data.Class>1.0

data[filter1 & filter2]
data.loc[:,"V1"]

#data.iloc[:,2]
#rounding values.

#DataFrame.round(self, decimals=0, *args, **kwargs)



decimals = pd.Series([0,1, 2, 3], index=['time','v1', 'v2','v3'])

data.round(decimals)
data.V1.plot(kind="line",color="red",label="V1")

data.V2.plot(kind="line",color="blue",label="V2",alpha=.5)

plt.legend()

plt.title("Line Plot")

plt.show()





data.sum(axis=0)
np.mean(data.V1) # mean of V1
data1= data.head()

data2= data.tail()



data_concat= pd.concat([data1,data2],axis=0)

data_concat
