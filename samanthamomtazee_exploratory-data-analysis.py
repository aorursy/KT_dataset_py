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

PS4_GamesSales = pd.read_csv("../input/videogames-sales-dataset/PS4_GamesSales.csv" ,encoding = "latin-1")

Video_Games_Sales_as_at_22_Dec_2016 = pd.read_csv("../input/videogames-sales-dataset/Video_Games_Sales_as_at_22_Dec_2016.csv",encoding = "latin-1")

XboxOne_GameSales = pd.read_csv("../input/videogames-sales-dataset/XboxOne_GameSales.csv",encoding = "latin-1")
for x in XboxOne_GameSales["Year"]:

    print(x)
years = []



for x in XboxOne_GameSales["Year"].iteritems():

    temp = x[1]

    if  np.isnan(temp):

        continue

    else:

        years.append(temp)

        

#print(years)
XboxOne_GameSales.describe()
import seaborn as sns

sns.boxplot(x= "North America", y = "Genre", data = XboxOne_GameSales)
genre = XboxOne_GameSales["Genre"]



y_axis = genre.value_counts()

x_axis = genre.value_counts().keys()
import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot( x = y_axis, y = x_axis)

plt.xlabel("# of games")

plt.ylabel("Genre")
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

sns.distplot(years);
# Libraries

from wordcloud import WordCloud

import matplotlib.pyplot as plt

 

# Create a list of word

listing_genre = ""

for items in XboxOne_GameSales["Genre"].iteritems():

    listing_genre += str(items[1]) + " "

 

# Create the wordcloud object

wordcloud = WordCloud(width=600, height=600, margin=0).generate(listing_genre)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()