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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





df_playstore_data = pd.read_csv("../input/googleplaystore.csv")

df_playstore_data.head(3)

df_playstore_data.shape
df_playstore_data.dtypes
df_playstore_data.isna().sum()
%%time

pd.to_numeric(df_playstore_data.Reviews,errors="coerce")
%%time

df_playstore_data['Reviews_Float'] = df_playstore_data['Reviews'].apply(lambda x: float(x) if x.isnumeric() else float('NaN'))
ratings_mean_by_type = df_playstore_data.groupby('Type').Rating.mean()

ratings_mean_by_type = ratings_mean_by_type.iloc[1:3]
num_ratings_by_type = df_playstore_data.groupby('Type').Rating.size()

num_ratings_by_type = num_ratings_by_type.iloc[1:3]

plt.subplot(211)

plt.plot(ratings_mean_by_type)

plt.subplot(212)

plt.plot(num_ratings_by_type)#number of ratings for Free vs number of ratings for Paid
ratings_by_type =  df_playstore_data.groupby('Type').Rating.plot(y='Ratings',x='Number Of Ratings',ylim=(0,5),figsize=(12,4),legend=True,grid=1)

ratings_by_type.iloc[1:3]


ratings_by_reviews = df_playstore_data.groupby('Rating').Reviews_Float.plot(figsize=(12,4),grid=1)

ratings_by_reviews