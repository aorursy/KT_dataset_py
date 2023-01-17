# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Seaborn graphing

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/us-mean-height/mean_height_inches.csv", index_col=0)

# Pop just the latest data out 

data = data.drop(axis = 1, columns=[ "1999-2000","2001-2002","2003-2004","2005-2006","2007-2008","2009-2010","2011-2012","2013–2014" ])  



# Concerning 2015-2016

y = []

x = list(data.index)

for idx, val in enumerate(x):

    #print(idx, val, str(''.join(data.values[idx]).split(' ')[0]))

    height_inches = float(''.join(data.values[idx]).split(' ')[0])

    if "Men" in val:

        ibw = (50 + 2.3 * (height_inches - 60))

    else:

        ibw = (45 + 2.3 * (height_inches - 60))



    # IBW x 6 mL/k

    gtv = ibw * 6

    y.append(gtv)

    #print(val,height_inches,ibw,gtv)

    

plt.figure(figsize=(8,8))

# plotting strip plot with seaborn 

ax = sns.barplot(x, y,)



# giving labels to x-axis and y-axis 

ax.set(ylabel ='mL/k')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# Predicted Body Weight and Tidal Volume Chart Male

y_male = [((50 + 2.3 * (i - 60)) * 6) for i in range(58,78)]

x_male = [i for i in range(58,78)]

ax_2 = sns.barplot(x_male, y_male)

ax_2.set(ylabel ='mL/k', xlabel = 'inches')

# Predicted Body Weight and Tidal Volume Chart Female

y_female = [((45 + 2.3 * (i - 60)) * 6) for i in range(55,75)]

x_female = [i for i in range(55,75)]

ax_3 = sns.barplot(x_female, y_female)

ax_3.set(ylabel ='mL/k', xlabel = 'inches')
