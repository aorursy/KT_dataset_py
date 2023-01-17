# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cwurData.csv')
df.info()
df.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True ,linewidths=.5, fmt='.3f',ax=ax)

plt.show()
df.head(15)
df.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

df.publications.plot(kind = 'line', color = 'y',label = 'publications',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.citations.plot(color = 'r',label = 'citations',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')

df.quality_of_faculty.plot(color = 'g',label = 'quality_of_faculty',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper left')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

# Scatter Plot 

# x = publications, y = citations

df.plot(kind='scatter', x='publications', y='citations',alpha = 0.5,color = 'red')

plt.xlabel('publications')              # label = name of label

plt.ylabel('citations')

plt.title('publications citations Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

df.publications.plot(kind = 'hist',bins = 50,figsize = (12,8))

plt.show()
# clf() = cleans it up again you can start a fresh

df.publications.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
df=pd.read_csv('../input/cwurData.csv')
series = df['publications']        # data['Defense'] = series

print(type(series))

data_frame = df[['citations']]  # data[['Defense']] = data frame

print(type(data_frame))
# Comparison operator

print(4 > 1)

print(3!=2)

print(2!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = df['publications']>995     

df[x]
# 2 - Filtering pandas with logical_and



df[np.logical_and(df['publications']<5, df['citations']<5 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

df[(df['publications']<7) & (df['citations']<4)]
# For pandas we can achieve index and value

for index,value in df[['publications']][0:1].iterrows():

    print(index," : ",value)