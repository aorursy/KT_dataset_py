# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")

df.head()
l=list(df['amenities'])

#for demonstration, reducing data to 100 rows

l=l[:100]#remove this line to include whole data

l[:5]
l=[[word.strip('[" ]') for word in row[1:-1].split(',')] for row in l] #trimming { } symbols and splitting at ',' and trimming " from each word

l[:5]
cols=set(word for row in l  for word in row) #forming a set of distinct text

cols.remove('')

cols[:10]
new_df=pd.DataFrame(columns=cols)

new_df
for row_idx in range(len(l)):

    for col in cols:

        new_df.loc[row_idx,col]=int(col in l[row_idx])

new_df
#finally the entire code is given below

df=pd.read_csv("../input/train.csv") #load your dataset into df

l=[[word.strip('[" ]') for word in row[1:-1].split(',')] for row in list(df['amenities'])]

#for demonstration, reducing data to 100 rows

l=l[:100]#remove this line to include whole data

cols=set(word for row in l  for word in row)

cols.remove('')

print(cols)

new_df=pd.DataFrame(columns=cols)

for row_idx in range(len(l)):

    for col in cols:

        new_df.loc[row_idx,col]=int(col in l[row_idx])

new_df