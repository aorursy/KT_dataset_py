# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# style.use("fivethirtyeight")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pwd
movie_rating = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv")

df = pd.DataFrame(movie_rating)
df
df.head()

df.tail(3)
df.head(2)
df
%matplotlib inline
df.head(100).plot(x= 'Rating', y= 'Votes')
# %matplotlib inline
df.head(20).plot(x = "Metascore" , y =  "Revenue (Millions)")
df[df['Rating']==9]
(df["Revenue (Millions)"].value_counts()/6).plot(kind = 'bar')
a = df['Rating']
b = df["Votes"]
plt.plot(a,b , label = "Ratings")

c = df['Revenue (Millions)']
plt.plot(a, c, label = 'Revenue')
plt.legend()

plt.show()
movie_rating['Title'].value_counts().head(15).plot(kind = 'bar', figsize= (15,4))
movie_rating['Title'].value_counts().head(15).plot(kind = 'pie', figsize= (15,4))
movie_rating.iloc[:5,:2]

plt.plot(movie_rating)
plt.show()
