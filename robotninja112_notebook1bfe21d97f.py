# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

pd.set_option('display.max_rows', 20)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv("../input/winemag-data-130k-v2.csv", sep=",", decimal=",", engine='python')



# Any results you write to the current directory are saved as output.
df[df.price <= 150].price.hist(bins=15,grid=False,rwidth=0.9)
df.points.hist(bins=20,grid=False,rwidth=0.9)
mean_points = df.groupby('country', as_index=False)['points'].mean()

mean_price = df.groupby('country', as_index=False)['price'].mean()

df2 = pd.merge(mean_points,mean_price,on="country")



df2.plot.scatter("points", "price",s=50,alpha=0.75);
mean_points.sort_values(by="points").sort_values(by='points',ascending=False).reset_index(drop=True)
mean_price.sort_values(by="price").sort_values(by='price',ascending=False).reset_index(drop=True)
count = df["variety"].value_counts()

df[df['variety'].isin(count[count >= 5].index)].groupby('variety', as_index=False)['points','price'].mean().sort_values(by='points',ascending=False).reset_index(drop=True)