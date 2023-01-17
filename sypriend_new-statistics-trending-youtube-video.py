# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/DEvideos.csv")
views = data[["views"]]/10000
data.info()
data.head(25)
data.columns
data.corr()

data_id = data.category_id
data_views = data.views
data_likes = data.likes
data_dislikes = data.dislikes
best_views = data_views > 1000000
dislikes_views = data_dislikes > 5000
likes_views = data_likes > 100000


data.plot(kind = "scatter",x ="likes",y="views",color="blue",alpha= 0.25,grid=True)
plt.xlabel("likes")
plt.ylabel("views")
plt.title("Likes And Views Correlation Scatter Diagram")
plt.show()
data.plot(kind = "scatter",x ="dislikes",y="views",color="g",alpha= 0.25,grid=True)
plt.xlabel("dislikes")
plt.ylabel("views")
plt.title("Dislikes And Views Correlation Scatter Diagram")
plt.show()
video_error_or_removed = data[["video_error_or_removed"]]
views.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
views = data[["views"]]/1000


