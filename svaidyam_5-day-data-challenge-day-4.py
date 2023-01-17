# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



#plt.rcdefaults()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/database.csv", low_memory=False)

#print(df.dtypes)



df['Aircraft Damage'].fillna(0, inplace=True)

df1 = df[df['Aircraft Damage'] != 0]



species = df1.groupby('Species Name')

damages = species['Aircraft Damage'].agg([np.sum])

#print(damages)



positions = list(range(len(damages)))



# get the value for the heigth of each bar (# of bird incidents)

values = damages.values



# get the name of every row (in this case the bird species)

labels = damages.index



# make a bar plot

# plot just the bars

plt.bar(positions, values)

# add the labels

plt.xticks(positions, labels)



plt.show()


