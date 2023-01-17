# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
man_df=pd.read_csv('../input/man_07032561.csv')

man_df['score']=man_df['score'].fillna(0)

fig = plt.figure(figsize = (8,6))

ax = fig.gca()

man_df['score'].hist(ax=ax,bins=30)

plt.axvline(man_df['score'].mean(), color='k', linestyle='dashed', linewidth=1)

plt.axvline(130,color='r',linewidth=1)

plt.axvline(man_df['score'].median(),color='b', linestyle='dashed',linewidth=1)

_, max_ = plt.ylim()

plt.text(man_df['score'].mean() + man_df['score'].mean()/10, 

         max_/2, 

         'Mean: {:.2f}'.format(man_df['score'].mean()))

plt.text(130+130/10,max_/3,'My Score now : 130')

plt.text(man_df['score'].median()*11/10,max_*9/10,'Median : {:.2f}'.format(man_df['score'].median()))

plt.title('Man score\n')

plt.ylabel('count')

plt.xlabel('score')

plt.show()
woman_df=pd.read_csv('../input/woman_01032561.csv')

woman_df['score']=woman_df['score'].fillna(0)

fig = plt.figure(figsize = (8,6))

ax = fig.gca()

woman_df['score'].hist(ax=ax,bins=30)

plt.axvline(woman_df['score'].mean(), color='k', linestyle='dashed', linewidth=1)

_, max_ = plt.ylim()

plt.text(woman_df['score'].mean() + woman_df['score'].mean()/10, 

         max_ - max_/10, 

         'Mean: {:.2f}'.format(woman_df['score'].mean()))

plt.title('Woman Score\n')

plt.xlabel('score')

plt.ylabel('count')

plt.show()