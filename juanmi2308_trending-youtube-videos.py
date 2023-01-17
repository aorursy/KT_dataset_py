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

import matplotlib as plt



df = pd.read_csv("../input/USvideos.csv")

df



NFL = df[df["channel_title"]=="NFL"]

NFL

df2 = df[df["title"]=="Cowboys vs. Falcons | NFL Week 10 Game Highlights"]

df2



g1 = df2.plot.bar(x="title", y="views", rot=0)
df3= df[df["title"]=="Cowboys vs. Falcons | NFL Week 10 Game Highlights"]

df3



df3 = pd.DataFrame({'likes': df3["likes"],

                   'dislikes': df3["dislikes"],

                   'comments': df3["comment_count"]})

                                            

ax = df3.plot.bar(rot=0)
dfa= df[df["title"]=="Cowboys vs. Falcons | NFL Week 10 Game Highlights"]

dfa['interactions'] = dfa['comment_count'] + dfa["likes"] + dfa["dislikes"]

dfa

df4= df[df["title"]=="Cowboys vs. Falcons | NFL Week 10 Game Highlights"]

df4





df9 = pd.DataFrame({'likes': df4["likes"],

                   'dislikes': df4["dislikes"],

                   'comments': df4["comment_count"],

                   'interactions': dfa["interactions"]})

                                            

g3 = df9.plot.bar(rot=0)
df10 = pd.DataFrame({'Views': df4["views"],

                   'interactions': dfa["interactions"]})

                                            

g3 = df10.plot.bar(rot=0)
Likes = df4["likes"].sum()

Dislikes = df4["dislikes"].sum()

Comment_count = df4["comment_count"].sum()



df11 = pd.DataFrame({'Interactions': [Likes, Dislikes, Comment_count]},

                    index=['Likes', 'Dislikes', 'Comment_count'])

plot = df11.plot.pie(y='Interactions', figsize=(5, 5))
from sklearn.tree import DecisionTreeRegressor



y = df4.views

features = ["likes", "dislikes", "comment_count", "category_id"]

X = df4[features]



views_model = DecisionTreeRegressor(random_state=1)

views_model.fit(X, y)

predictions = views_model.predict(X)

print(predictions)



df4

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

val_predictions = views_model.predict(val_X)

print(val_predictions)



from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_predictions, val_y)

print(val_mae)

print(predictions)