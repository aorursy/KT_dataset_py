# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.




X_train = pd.read_json("../input/train.json")

X_test = pd.read_json("../input/test.json")



interest_level_map = {'low': 0, 'medium': 1, 'high': 2}

X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])

X_test['interest_level'] = -1



X_train.head()

X_test.head()
high_df=X_train[X_train['interest_level']==2]

medium_df=X_train[X_train['interest_level']==1]

low_df=X_train[X_train['interest_level']==0]
print(high_df.head())

print(medium_df.head())

print(low_df.head())
print(len(high_df))

print(len(medium_df))

print(len(low_df))
print(high_df['bedrooms'].min()) #minimum number of bedrooms for high interest level

print(medium_df['bedrooms'].min())

print(low_df['bedrooms'].min())
print(high_df['bedrooms'].max())#maximum number of berooms for high interest level

print(medium_df['bedrooms'].max())

print(low_df['bedrooms'].max())
print(high_df['bedrooms'].mean()) # average number of bedrooms for high interest level

print(medium_df['bedrooms'].mean())

print(low_df['bedrooms'].mean())
print(high_df['bathrooms'].min()) 

print(medium_df['bathrooms'].min()) 

print(low_df['bathrooms'].min()) 
print(high_df['bathrooms'].max())

print(medium_df['bathrooms'].max())

print(low_df['bathrooms'].max())
print(high_df['bathrooms'].mean())

print(medium_df['bathrooms'].mean())

print(low_df['bathrooms'].mean())
print(high_df['price'].min())

print(medium_df['price'].min())

print(low_df['price'].min())
print(high_df['price'].max())

print(medium_df['price'].max())

print(low_df['price'].max())
print(high_df['price'].mean())

print(medium_df['price'].mean())

print(low_df['price'].mean())
print(high_df['created'].min())

print(medium_df['created'].min())

print(low_df['created'].min())
print(high_df['created'].max())

print(medium_df['created'].max())

print(low_df['created'].max())