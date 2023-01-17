# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting

import seaborn as sea #for visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing datasets 



movies_df = pd.read_csv("../input/IMDB-Movie-Data.csv")
#show the columns



movies_df.columns
# there was a blank in some columns. We removed them

movies_df.columns=[i.split()[0]+"_"+i.split()[1]  if len(i.split())>1 else i for i in movies_df.columns]



# and remove paranthesis

movies_df=movies_df.rename(columns = {'Revenue_(Millions)':'Revenue_Millions'})

movies_df=movies_df.rename(columns = {'Runtime_(Minutes)':'Runtime_Minutes'})



movies_df.columns

#after importing, a glimpse at movies_df



movies_df.head()
#we have 3 float columns, 4 integer and 5 object columns according to info() method



movies_df.info()
#some numeric informations about the movies_df



movies_df.describe()
#try make a correlation map with using seaborn lib.



movies_corr = movies_df.corr()

f,ax = plt.subplots(figsize=(10, 10))

sea.heatmap(movies_corr, annot = True, linewidths = 0.1, fmt= '.2f', ax=ax )

plt.show()
# these are the rating point in the database



print("Rating Points :",movies_df['Rating'].unique())
# lets see how many films are there for each rating point



print(movies_df['Rating'].value_counts())

# lets visualize rating points with pie chart



plt.figure(1, figsize=(10,10))

movies_df['Rating'].value_counts().plot.pie(autopct="%1.1f%%")
#scatter plot about movie and their ratings between 2006 - 2016



plt.scatter(movies_df.Year, movies_df.Rating, alpha = 0.28, label = "Movie", color = "blue")

plt.xlabel("Years")

plt.ylabel("Ratings")

plt.legend(loc = "lower right")

plt.show()
#histogram plot about number of published movies according to year



movies_df.Year.plot(kind = "hist", bins = 40, figsize = (9,6))

plt.xlabel("Years")

plt.ylabel("Number of Movies")

plt.show()

movies_df["Runtime_Minutes"].value_counts()
movies_df.Runtime_Minutes.value_counts()[:10].plot.pie(autopct='%1.1f%%',figsize=(10,10))

plt.title('Top 10 runtime of Movies')
movies_time=movies_df.Runtime_Minutes

f,ax = plt.subplots(figsize=(14, 8))

sea.distplot(movies_time, bins=20, kde=False,rug=True, ax=ax);

plt.ylabel("Counts")