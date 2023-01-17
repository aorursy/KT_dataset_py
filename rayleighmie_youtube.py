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
import pandas as pd

import codecs

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")

FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")

GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")

INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")



with codecs.open("../input/youtube-new/JPvideos.csv", "r", "UTF8", "ignore") as file:

    JPvideos = pd.read_table(file, delimiter=",")

with codecs.open("../input/youtube-new/KRvideos.csv", "r", "UTF8", "ignore") as file:

    KRvideos = pd.read_table(file, delimiter=",")

with codecs.open("../input/youtube-new/MXvideos.csv", "r", "UTF8", "ignore") as file:

    MXvideos = pd.read_table(file, delimiter=",")

with codecs.open("../input/youtube-new/RUvideos.csv", "r", "UTF8", "ignore") as file:

    RUvideos = pd.read_table(file, delimiter=",")
X = JPvideos.copy()

X = X.drop('video_id',axis=1)

X = X.drop('trending_date',axis=1)

X = X.drop('thumbnail_link',axis=1)

X = X.drop('comments_disabled',axis=1)

X = X.drop('video_error_or_removed',axis=1)

display(X.head())





print(JPvideos.shape)

y = JPvideos.copy().iloc[:, [7]]

display(y.head())



JPvideos.drop_duplicates()

print(JPvideos.shape)
def value(video_value):

 if row["ratings_disabled"]:

    return row["views"]

 else:

    return row["views"] * row["likes"] - row["dislikes"]



for video_value in X.iterrows():

    X['video_value'][index] = video_value

X_new = pd.concat([X, video_value], axis=1)

display(X_new.head(5))
def value(row):

 if row["ratings_disabled"]:

    return row["views"] * row["likes"] - row["dislikes"]

 else:

    return row["views"]

        

X.applymap(value)