import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns  #importing packages for visualization
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np #importing packages for handling data

df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv') #loading csv file from github
data=df.drop(['fips'], axis = 1) #dropping fips attribute from the data

data

data.sort_values(by=['cases'], ascending=False) ##arranging no of cases in descending order

data.sort_values(by=['deaths'], ascending=False) ## arranging deaths in descending order
plt.figure(figsize=(20,12)) # Figure size

data.groupby("state")['cases'].max().plot(kind='bar', color='yellow') ##plotting number of cases of covid-19 in different states 
data.plot.line() ##plotting the number of cases along with deaths in individual states 
plt.figure() # for defining figure sizes

data.plot(x='state', y='deaths', figsize=(20,12), color='red') ##plotting number of deatha in individual states