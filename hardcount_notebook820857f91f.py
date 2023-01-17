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
movies = pd.read_csv("../input/movie_metadata.csv")

movies.head(4)
#fill 0 values for gross and budget which currently have NaN

movies = movies.fillna({"gross":0, "budget":0})
#format gross and budget as currency

usd = lambda x: '${:,.2f}'.format(x)
#calucate profit, gross - budget

movies["profit"] = movies["gross"] - movies["budget"]
#this statement creates a new dataframe of relivant columns and displays the 3 most profitable movies

most_profitable = movies[["movie_title", "gross", "budget", "profit"]].sort_values("profit", ascending=False).head(3)
#format columns as $USD

most_profitable["gross"] = most_profitable["gross"].map(usd)

most_profitable["budget"] = most_profitable["budget"].map(usd)

most_profitable["profit"] = most_profitable["profit"].map(usd)



most_profitable
#find top 5 most liked non-english speaking movie, display name, director, language, and movie_facebook_likes
movies[["movie_title", "director_name", "language", "movie_facebook_likes"]][movies["language"]!="English"].sort_values("movie_facebook_likes", ascending=False).head(5)
#Aggregate total gross, average gross, and number for movies for all directors
aggregate = { 

    "gross": {

        "total_gross":"sum",

        "average_gross":"mean",

        "number_of_movies":"count"

    }    

}
movies.groupby("director_name").agg(aggregate)