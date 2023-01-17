# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

data.head()
new_data = data.drop([

                    "thumbnail_link","video_id","trending_date","publish_time",

                    "comments_disabled","video_error_or_removed","ratings_disabled"],

                     axis = 1)



new_data.head()
# Indexes and Columns in our new_data 



print(new_data.index,"---",new_data.columns)
high = new_data["views"].max()  # High of views Videos

lower = new_data["views"].min() # Minimal value of views Videos 



highest = new_data[new_data["views"] == high ]

lowest = new_data[new_data["views"] == lower ]



print("""Name of video of highest views: "{}" and views is : {}""".format(highest["title"].iloc[0],highest["views"].iloc[0]))

print("""Name of video of lowest views: "{}" and views is : {}""".format(lowest["title"].iloc[0],lowest["views"].iloc[0]))
# First named the index of new_data

new_data.index.names = ["indexes"]

new_data.plot( kind="scatter",x="category_id",y="likes",figsize=(10,10))

plt.xlabel("Video categories")

plt.ylabel("Values of likes")

plt.title("Likes of category")

plt.show()
new_data.plot( kind="scatter",x="category_id",y="views",color="red",figsize=(10,10))

plt.xlabel("Video categories")

plt.ylabel("Values of views")

plt.title("Views with category")

plt.show()
new_data[["category_id","comment_count"]].groupby("category_id").sum() # Group by category id

new_data[["category_id","comment_count"]].groupby("category_id").mean() # mean of comment count with category id
videos_no = new_data["category_id"].value_counts() # Number of videos ( every category have one video)



videos_no.sort_index()
def number_of_tags(data):

    

    tags = data.split("|") # The Tags for each videos is with "|" divided

    

    return len(tags)  # Number of Tags



new_data["tags_number"] = new_data["tags"].apply(number_of_tags)



new_data.head()
def ratio_likes_dislikes(likes,dislikes):

    

    likes_values = np.array([])    # Make ampty array of the likes values

    dislikes_values = np.array([]) # Make ampty array of the dislikes values

    

    for values in likes:

        

        likes_values = np.append(likes_values,values)   # Update the empty likes_values with the values of likes with numpy append() 

        

    for values in dislikes:

        

        dislikes_values = np.append(dislikes_values,values) # Update the empty dislikes_values  with the values of dislikes with numpy.append()

    

          

    ratio_likes_dislikes = np.array([]) # Calculate rate of likes and dislikes

    

    ratio_likes_dislikes = np.append(ratio_likes_dislikes,likes_values / (likes_values + dislikes_values))

    

    ratio_likes_dislikes[np.isnan(ratio_likes_dislikes)] = 0

    

    return ratio_likes_dislikes



likes = new_data["likes"]

dislikes = new_data["dislikes"]



new_data["likes_dislikes"] = ratio_likes_dislikes(likes,dislikes)



new_data.head()
# Now sort values by likes_dislikes from highest to lowest 



new_data.sort_values(by = "likes_dislikes",ascending = False,inplace = True)



new_data.head(10)
#Plotting graph of category id with likes_dislikes 



new_data.plot( kind="scatter",x="category_id",y="likes_dislikes",figsize=(10,10))

plt.xlabel("Video categories")

plt.ylabel("Values of likes_dislikes")

plt.title("Likes_dislikes with category")

plt.show()
new_data["likes_dislikes"].plot(kind="hist",color="g",figsize=(10,10))

plt.xlabel("likes_dislikes")

plt.show()