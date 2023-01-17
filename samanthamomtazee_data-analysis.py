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
top10s = pd.read_csv("../input/top-spotify-songs-from-20102019-by-year/top10s.csv", encoding = "latin-1")
top10s.head(300)
print(top10s["pop"])
# library

import matplotlib.pyplot as plt

 

# library & dataset

import seaborn as sns

 

# use the function regplot to make a scatterplot

#sns.regplot(x=df["sepal_length"], y=df["sepal_width"])

#sns.plt.show()

 

# Without regression fit:

sns.regplot(x=top10s["pop"], y=top10s["bpm"], fit_reg=False)

plt.show()





# Libraries

from wordcloud import WordCloud

import matplotlib.pyplot as plt

 

# Create a list of word

text=(str(top10s["title"]))

# Create the wordcloud object

wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)

 

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
