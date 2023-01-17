# Importing all packages and setting plots to be embedded inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

import statsmodels.api as sm

from sklearn.cluster import KMeans



%matplotlib inline
# Loading the dataset into a pandas dataframe

df=pd.read_csv('../input/1987_modified.csv')
plt.figure(figsize=[8,5])

bins=np.arange(0,df['Total_Delay'].max()+10,25)

plt.hist(data=df,x='Total_Delay',bins=bins,rwidth=0.7)

plt.title('Distribution of Total delay')

plt.xlabel('Delay duration in minutes')

plt.xlim(0,400)

plt.xticks([0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375]);
plt.figure(figsize=[8,5])

sb.barplot(data=df,x='UniqueCarrier',y='Total_Delay')

plt.title('Unique Carrier vs Total delay')

plt.xlabel('Carrier Code')

plt.ylabel('Count of Delays');
g=sb.FacetGrid(data=df,col='Month',margin_titles=True,sharex=False,sharey=False,height=5)

g.map(sb.barplot,'UniqueCarrier','Total_Delay')

g.add_legend();
plt.figure(figsize=[8,5])

sb.regplot(data=df,x='Total_Delay',y='Distance',fit_reg=False)

plt.title('Distance vs Total delay')

plt.xlabel('Total delay in minutes')

plt.ylabel('Distance in miles');
sb.regplot(data=df,y='Distance',x='ActualElapsedTime',fit_reg=False)

plt.title('Distance vs Actual elapsed time')

plt.xlabel('Actual elapsed time in minutes')

plt.ylabel('Distance in miles');
g=sb.FacetGrid(data=df,col='Month',margin_titles=True,sharex=False,sharey=False)

g.map(plt.scatter,'Total_Delay','Distance')

g.add_legend();