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
df = pd.read_csv('../input/berlin-airbnb-data/listings.csv')
df.head()
df_clean = df[['price', 'minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
df_clean.head()
df_clean.info()
df_clean.dropna(inplace=True)
df_clean.info()
df_clean.price.hist()
df_clean[df_clean['price'] < 300].price.hist(bins=100)
df_clean.price.min()
df_clean.price.max()
import pandas as pd

df_Credit_Card = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
df_Credit_Card.head()
df_Credit_Card.drop(['ID'], axis=1, inplace=True)
df_Credit_Card.hist(figsize=(15,15))
df_Credit_Card