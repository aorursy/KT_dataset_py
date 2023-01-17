# import modules we'll need

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# read in people csv & peek at the first few lines

people = pd.read_csv("../input/SDFB_people.csv")

people.head()
# make sure the year of birth & year of death both exist

people["Extant Birth Year"].isnull().sum() == 0 & people["Extant Death Year"].isnull().sum() == 0



# unfortunantly these rows jave a not a number death year in them

print(people[people["Extant Death Year"]=="1710/11"])

print(people[people["Extant Death Year"]=="1738/10/12"])



# get the rows that aren't that one

people = people[people["Extant Death Year"]!="1710/11"]

people = people[people["Extant Death Year"]!="1738/10/12"]



# and calculate their ages

ages = pd.to_numeric(people["Extant Death Year"]) - pd.to_numeric(people["Extant Birth Year"])
ageRange = plt.hist(ages, bins='auto')

plt.title("Histogram of Individuals' Ages")

plt.show
# let's make a wordcloud



# import the modules we'll need

from scipy.misc import imread

import random

from wordcloud import WordCloud, STOPWORDS



# get all the text that is not null

histSign = people["Historical Significance"][-pd.isnull(people["Historical Significance"])]

text = ' '.join(histSign)



# make a wordcloud

wordcloud = WordCloud(relative_scaling = 1.0).generate(text)

plt.imshow(wordcloud)

plt.axis("off")

plt.show()