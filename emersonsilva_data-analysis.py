#import libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



colormap = plt.cm.viridis
#verifying if has missing data

df_impacts = pd.read_csv('../input/impacts.csv')

df_impacts.empty
#check out maximun value of Torino Scale

torino_scale = df_impacts['Maximum Torino Scale']

print(torino_scale.max())



names = df_impacts['Object Name']



df_impacts.drop(['Object Name','Maximum Torino Scale','Asteroid Magnitude'], axis=1, inplace=True)



#calculating the complete period for each asteroid

df_impacts['Period'] = df_impacts['Period End']-df_impacts['Period Start']

df_impacts.drop(['Period End', 'Period Start','Maximum Palermo Scale'], axis=1, inplace=True)
df_impacts.describe()
#looking correlation between variables

sns.heatmap(df_impacts.corr(), annot=True, cmap=colormap)
x = df_impacts['Period']

y = df_impacts['Possible Impacts']

sns.regplot(x,y)
sns.regplot(df_impacts['Period'],df_impacts['Cumulative Palermo Scale'])
sns.regplot(df_impacts['Asteroid Diameter (km)'],df_impacts['Cumulative Palermo Scale'])
for i in df_impacts.columns:

    sns.boxplot(df_impacts[i])

    plt.show()

    plt.clf()

    sns.distplot(df_impacts[i])

    plt.show()

    plt.clf()
#the biggest asteroid

max_diam = df_impacts['Asteroid Diameter (km)'].max()



j = 0

for i in df_impacts['Asteroid Diameter (km)']:

    if i == max_diam:

        break

    j += 1



print(names[j])

df_impacts.loc[j]
#the asteroid with most impact probability

max_prob = df_impacts['Cumulative Impact Probability'].max()



j = 0

for i in df_impacts['Cumulative Impact Probability']:

    if i == max_prob:

        break

    j += 1



print(names[j])    

df_impacts.loc[j]
#the asteroid with higher number of possible impacts

max_impacts = df_impacts['Possible Impacts'].max()



j = 0

for i in df_impacts['Possible Impacts']:

    if i == max_impacts:

        break

    j += 1



print(names[j])    

df_impacts.loc[j]