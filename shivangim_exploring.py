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
print(list(frame.columns.values))
frame = pd.read_csv("../input/movie_metadata.csv")

frame["MovieID"] = frame.index
print("Shape : " + str(frame.shape))

print("Size :  " + str(frame.size))

frame.head()
frame["genres"][0].split("|")

unique_genres = []

for i in range(0, 5043):

    g = frame["genres"][i].split("|")

    for i in g:

        if i not in unique_genres:

            unique_genres.append(i)

print(len(unique_genres))

print(unique_genres)
rating_count = pd.DataFrame(frame.groupby('MovieID')['imdb_score'].sum())

#print rating_count

rating_count.sort_values('imdb_score', ascending=False).head()
frame["imdb_score"].describe()