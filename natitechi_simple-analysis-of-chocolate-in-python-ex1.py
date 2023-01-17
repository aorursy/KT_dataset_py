import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# use seaborn default plotting style

sns.set()
# read the data

df = pd.read_csv('../input/flavors_of_cacao.csv')



# have a look

df.head()
# remove some unnecessary strings `\n` and `\xa0`

df.columns = df.columns.str.replace('\n', ' ').str.replace('\xa0', '')



df.columns.tolist()
# convert `string` to `float`

df['Cocoa Percent'] = df['Cocoa Percent'].apply(lambda row: row[:-1]).astype('float')



# check data types

df.dtypes
# look at correlation

df.corr()
# set marker size based on the cocoa percent of a chocolate

markerprop = {'s': .1*df['Cocoa Percent']**1.7, 'alpha': .4}



# scatterplot

ax = sns.lmplot(x='Broad Bean Origin', y='Rating', data=df,

                scatter_kws=markerprop, 

                fit_reg=False, legend=False, aspect=2.5)



# add averaged rating of each `Bean Origin`

df_copy = df.copy().groupby(['Broad Bean Origin'])['Rating'].mean()

plt.plot(df_copy, color='k', marker='o', alpha=.8, lw=3, label='Averaged Rating')



# rotate x tick label

ax.set_xticklabels(rotation=90)



# graph properties

plt.title('Rating of Chocoloate VS Bean Origin (2006-2017)', fontsize=20)

plt.xlabel('Broad Bean Origin', fontsize=15)

plt.ylabel('Rating (1-5)', fontsize=15)

plt.margins(.03)

plt.legend()



plt.show()
# get the cocoa beans with the highest average rating

df_copy[df_copy == df_copy.max()]
# set marker size based on the cocoa percent of a chocolate

markerprop = {'s': .1*df['Cocoa Percent']**1.7, 'alpha': .2, 'color': 'navy'}



# scatterplot

ax = sns.lmplot(x='Company Location', y='Rating', data=df,

                scatter_kws=markerprop, 

                fit_reg=False, legend=True, aspect=2.5)



# add averaged rating

df_copy = df.copy().groupby(['Company Location'])['Rating'].mean()

plt.plot(df_copy, color='r', marker='o', alpha=.8, lw=3, label='Averaged Rating')



# # graph properties

ax.set_xticklabels(rotation=90)

plt.title('Rating of Chocoloate VS Company Location (2006-2017)', fontsize=20)

plt.xlabel('Company Location', fontsize=15)

plt.ylabel('Rating (1-5)', fontsize=15)

plt.margins(.03)



plt.show()
# get samples in Chile

df[df['Company Location'] == 'Chile']
# get samples in India

df[df['Company Location'] == 'India']