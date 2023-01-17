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
import pandas as pd



reviews = pd.read_csv("../input/ign.csv")

print ("ok")
reviews.head()
reviews.shape
reviews.iloc[0:5,:]
#remove first column

reviews = reviews.iloc[:,1:]

reviews.head()
reviews.loc[0:5,:]
reviews.index
some_reviews = reviews.iloc[10:20,]

some_reviews.head()
reviews.loc[:5,"score"]
reviews.loc[:5,["score", "release_year"]]
reviews[["score", "release_year"]]
type(reviews["score"])
reviews["score"].mean()
reviews.mean()
reviews.corr()
#show scores higher than 7

score_filter = reviews["score"] > 7

score_filter
#select rows where reviews are higher than 7

filtered_reviews = reviews[score_filter]

filtered_reviews.head()
#select xbox games where reviews are higher than 7

xbox_one_filter = (reviews["score"] > 7) & (reviews["platform"] == "Xbox One")

filtered_reviews = reviews[xbox_one_filter]

filtered_reviews.head()
#plot xbox one reviews

%matplotlib inline

reviews[reviews["platform"] == "Xbox One"]["score"].plot(kind="hist")
#plot PS4 reviews

reviews[reviews["platform"] == "PlayStation 4"]["score"].plot(kind="hist")
filtered_reviews["score"].hist()