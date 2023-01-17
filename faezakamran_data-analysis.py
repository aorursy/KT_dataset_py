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
df = pd.read_csv("../input/avocado.csv")

df.head() 



#you can specify the no of rows you want to print like this

#df.head(3)
df.tail(4)
df["AveragePrice"].head()
#if we need the information of a single region let's say, Albany

albany_df = df[ df['region'] == 'Albany'] 

albany_df.head()
#date seems to be the unique identifier here so setting it as index

albany_df = albany_df.set_index('Date')

albany_df.head()




albany_df.plot()
albany_df['AveragePrice'].plot()