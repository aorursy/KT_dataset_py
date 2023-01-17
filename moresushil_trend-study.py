import timeit

start = timeit.default_timer()

#Your statements here
# Imports for this project

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')

sns.set(font_scale=1.5)



# Read in TN middle school dataset from GitHub

df = pd.read_csv('https://raw.githubusercontent.com/LearnDataSci/article-resources/master/Essential%20Statistics/middle_tn_schools.csv')



df.describe()
df[['reduced_lunch', 'school_rating']].groupby(['school_rating']).describe()
# only view these two variables

df[['reduced_lunch', 'school_rating']].corr()
fig, ax = plt.subplots(figsize=(16,16))



ax.set_ylabel('school_rating')



# boxplot with only these two variables

_ = df[['reduced_lunch', 'school_rating']].boxplot(by='school_rating', figsize=(13,13), vert=False, sym='b.', ax=ax)
plt.figure(figsize=(16,16)) # set the size of the graph

_ = sns.regplot(data=df, x='reduced_lunch', y='school_rating')
# create tabular correlation matrix

corr = df.corr()

_, ax = plt.subplots(figsize=(13,13)) 



# graph correlation matrix

_ = sns.heatmap(corr, ax=ax,

                xticklabels=corr.columns.values,

                yticklabels=corr.columns.values,

                cmap='Blues')
stop = timeit.default_timer()



print('Time: ', stop - start)