# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output







# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tripadvisor_co_uk-travel_restaurant_reviews_sample.csv")

df = pd.DataFrame(data)

print("we have",df.shape[0],"rows and",df.shape[1],"columns")
df.head(10)
unique_name = df["name"].unique()

comb = df[["name",'restaurant_location']].groupby("name")

comm = comb['name']

new_name = df.drop_duplicates(['name'])
time = df["review_date"].head(10)
