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
data = pd.read_csv("../input/food-com-recipes-and-user-interactions/RAW_interactions.csv")

data.head()

data.shape
data.iloc[0:6,4:5] # for selecting useful columns. İnstead of 0:6 we can use [0,1,2,3,4,5] like data.iloc[[0,1,2,3,4,5],4:5]
data.loc[0:6,["review"]]  #label based selection
data.loc[data.rating == 5] # if data.rating was a string it should have be  "5" as a string format. And we can use & and | like data.loc[(data.rating == 5) & (data.date == "2011-12-21")]
data.loc[data.rating.isin([5,4])] # isin function selecting automatically instead of data.loc[(data.rating == 5) | data.loc[(data.rating == 4)]
data.rating.isnull() # or it's companion notnull for selecting not null 
data.rating.describe()
data.user_id.unique()
data.recipe_id.value_counts()
data.groupby("rating").rating.count()

# we can see how many of them.
data.groupby("recipe_id").rating.min()

data.loc[data.recipe_id.isin(["38"])] # recipe_id = 38 shows that the worst rating is 4 and .rating.min() works truely
#data.groupby(["date","rating"]).recipe_id.counts()
data.groupby(["recipe_id"]).rating.agg(['min','max','sum','size','mean'])
user_activity = data.groupby(['user_id', 'recipe_id']).rating.agg( len) # Multi 

user_activity

#user_activity.groupby(["user_id"]).recipe_id.agg([len])
user_activity.reset_index()
# this should work according to courses of kaggle idk why it is not working 

#user_activity.sort_values(by = ["rating"] , ascending=False)

#user_activity.sort_values(by=['country', 'len'])

data[pd.isnull(data.review)]

data.review.fillna("Unknown")

data.loc[data.review == "Unknown"] 

# the code above is not working I do not know why
data.review.replace("NaN","Unknown")

data.loc[data.review == "Unknown"] 

# the code above is not working I do not know why
datarenamed = data.rename(columns = {"user_id":"user"})

datarenamed = datarenamed.rename(columns = {"recipe_id":"recipe"})

datarenamed.head()
datarenamed = datarenamed.rename(index = {0:"comment1",1:"comment2"})

datarenamed.head(2)
