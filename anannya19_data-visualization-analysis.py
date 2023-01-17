#Step 1 : Load the file 

import numpy as np # for algebra
import pandas as pd # for data processing
import matplotlib.pyplot as plt
import seaborn as sns #for data visualization
import os


#data = pd.read_csv(path)
data = pd.read_csv("../input/countries of the world.csv")
#function : get the details of file
def basic_details(df):
    print('Row:{}, columns{}'.format(df.shape[0], df.shape[1]))
    k = pd.DataFrame()
    k['Number of Unique values'] = df.nunique()
    k['Number of Missing values'] = df.isnull().sum()
    k['Data Type'] = df.dtypes
    return k

#Step 3 : Call the function by passing in parameter : data

basic_details(data)
#Step 4 : graph the distribution by climate

plt.figure(figsize=(12,4))
ax = sns.countplot(data['Climate'])
plt.title('Distribution by climate');

#conclusion : most of countries have climate of 2
#Step 5 : #use a tree map to visualize size of number of countries to region

import squarify #...import this package to anaconda : pip install squarify (treemap algorithm)
plt.figure(figsize=(12,12))

by_region = data.groupby('Region')

a= by_region['Country'].count()

a.sort_values(ascending=False,inplace=True)

squarify.plot(sizes= a[0:15].values, label= a[0:15].index, alpha=0.9)

plt.axis('off')
plt.tight_layout()

#conclusion : largest size of countries is spread btw latin america & sub saharan africa
#Step 6 : Calculate correlation using statistical fn

data.corr()
#Step 7 : show the heatmap to demo correlation statistics

#show the heatmap
plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot= True, linewidths =0.5)
plt.show()

