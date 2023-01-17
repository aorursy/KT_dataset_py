# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df= pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

df.head(10)
df.columns
df.corr()
import matplotlib.pyplot as plt

df.release_year.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
x= df["country"]=="Turkey"

df[x]
print("threshold:",threshold)




threshold = sum(df.release_year)/len(df.release_year)

df["release_year_level"] = ["new" if i > threshold else "old" for i in df.release_year]

df.loc[:20,["release_year_level","release_year"]]

                      



df.info()
print(df['release_year'].value_counts(dropna =False))
df_new = df.head()

df_new
melted=pd.melt(frame=df_new, id_vars="type", value_vars=["release_year","duration"])

melted

              

df.dtypes
df.info()
df["director"].value_counts(dropna=False)