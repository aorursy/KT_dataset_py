# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



all_plays = pd.read_csv("../input/NFLPlaybyPlay2015.csv")



# Any results you write to the current directory are saved as output.
del all_plays["Unnamed: 0"]

all_plays.columns
down4_plays = all_plays[all_plays.down == 4]

down4_plays.PlayType.value_counts()
plt.rcParams['figure.figsize'] = (4, 3)

%config InlineBackend.figure_format='retina'



ax = plt.subplot(111)    

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)



ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left()

# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    

plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")    

  

down4_plays.PlayType.value_counts().plot(kind='bar')
down4_plays.head()
down4_plays[down4_plays.PlayType == "Pass"].plot(kind='scatter', x='TimeSecs', y='ydstogo')#, c='PlayType')
grp_d4 = down4_plays.groupby("qtr")
down4_plays.groupby("qtr").PlayType.value_counts().plot(kind='bar', alpha=0.7)

# .qtr.value_counts().sort_index()

# down4_plays[down4_plays.PlayType == "Run"].qtr.value_counts().sort_index().plot(kind='bar', color="#ff0000", alpha=0.6)
ax = plt.subplot(111)    

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)



ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left()

# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    

plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")  



down4_plays[down4_plays.qtr == 1].PlayType.value_counts().plot.bar(stacked=True, ax=ax, label='Q1', color="blue")

down4_plays[down4_plays.qtr == 2].PlayType.value_counts().plot.bar(stacked=True, ax=ax, label='Q2', color="red")

down4_plays[down4_plays.qtr == 3].PlayType.value_counts().plot.bar(stacked=True, ax=ax, label='Q3', color="green")

down4_plays[down4_plays.qtr == 4].PlayType.value_counts().plot.bar(stacked=True, ax=ax, label='Q4', color="yellow")

down4_plays[down4_plays.qtr == 5].PlayType.value_counts().plot.bar(stacked=True, ax=ax, label='OT', color="gray")



plt.legend()