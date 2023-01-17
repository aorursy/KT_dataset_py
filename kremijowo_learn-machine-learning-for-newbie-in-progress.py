import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 

import matplotlib.pyplot as plt 

import seaborn as sns



# check list of input in the directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/pokemon.csv')
data.info()
#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)
data.head(10)
data.columns
# LINE PLOT

data.Speed.plot( kind = 'line', color = 'g', label = 'Speed',

                linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.Defense.plot( kind = 'line', color = 'r', label = 'Defense',

                linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')



plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')



# Scatter Plot

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',

          alpha=0.5, color = 'green')

plt.labelx=('Attack')

plt.labely=('Defense')

plt.title=('Attack-Defense Scatter Plot')