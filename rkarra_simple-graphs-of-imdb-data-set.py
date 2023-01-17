# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/IMDB-Movie-Data.csv"]).decode("utf8"))

file = pd.read_csv('../input/IMDB-Movie-Data.csv')

print ("Hightest rating is",max(file.Rating), '',"Least Rating is",min(file.Rating))

#Movies with ratings greater than 6

print ("Movies with rating greater than 6.0 are"'\n',file.Title[file.Rating>6.0])





#GRAPH OF RATINGS

plt.hist(file.Rating)

plt.title('Ratings')

plt.show()

#Runtime of movies

plt.hist(file['Runtime (Minutes)'])

plt.title('Movie Runtime in Minutes')

plt.xlabel('Run time in minutes')

plt.ylabel('Number of movies in that run time')

plt.show()



# Remember drop na drops the empty values

rev=file['Revenue (Millions)'].dropna()

plt.title('Revenue')

plt.hist(rev)

plt.show()

#

## Metascore

met=file.Metascore.dropna()

plt.title('Metascore')

plt.hist(met)

plt.show()

# ###

#Movies by years

plt.title("Number of movies over the years")

plt.xticks(range(2005,2019),rotation=90)

plt.hist(file.Year,30)

plt.show()