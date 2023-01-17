import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
df = pd.read_csv('../input/movie_metadata.csv')

df.head()
# checking the director/movie with the biggest grossing

dfD=df.loc[:,['director_name','gross','movie_title']]

dfD.head()
# it's kinda hard to read so we will check only millions, and replace the nan

dfD['gross']=dfD['gross'].fillna(0)

dfD['grossMil']=dfD['gross']/1000000

dfD.sort_values(by="grossMil", ascending=False).head()
sorted_dfD=dfD.sort_values(by="grossMil")

plt.plot(sorted_dfD['grossMil'])

# there a lot with 0 we need to clean them
dfD