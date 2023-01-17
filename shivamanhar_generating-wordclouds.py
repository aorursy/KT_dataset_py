# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load in the dataframe

df = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
df.head(3)
print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))

print("There are {} types of wine in this dataset such as {} .. \n".format(len(df.variety.unique()),", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {} ..\n".format(len(df.country.unique()),",".join(df.country.unique()[0:5])))
df[["country","description", "points"]].head(5)
# Groupby by country



country = df.groupby("country")



#Summary statistic of all countries

country.describe().head()
country.mean().sort_values(by="points", ascending=False).head()
plt.figure(figsize=(15,10))

country.size().sort_values(ascending=False).plot.bar()

plt.xticks(rotation=50)

plt.xlabel("Country of Origin")

plt.ylabel("Number of Wines")

plt.show()
#country.size()
plt.figure(figsize=(15, 10))

country.max().sort_values(by="points", ascending=False)["points"].plot.bar()

plt.xticks(rotation =50)

plt.xlabel("Country of Origin")

plt.ylabel("Highest point of Wines")

plt.show()
#start with one review:

text = df.description[0]



#Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



#Display the Generated image:

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words =100, 

                      background_color='white').generate(text)

plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud.to_file("first_review.png")
## Combine all wine reviews into one big text

text = " ".join(review for review in df.description)

print("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:

stopwords  = set(STOPWORDS)

stopwords.update(["drink", "now", "wine", "flavor", "flavors"])



#Generate a word cloud image

wordcloud = WordCloud(stopwords = stopwords, background_color="white").generate(text)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wine_mask = np.array(Image.open("../input/images/wine_mask.png"))

wine_mask
# function to swap number 0 to 255

def transform_format(val):

    if val == 0:

        return 255

    else:

        return val
# Transform your mask into a new one that will work with the function:

transformed_wine_mask = np.ndarray((wine_mask.shape[0], wine_mask.shape[1]),

                                   np.int32)



for i in range(len(wine_mask)):

    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
# check the result 

transformed_wine_mask
wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask

              , stopwords =stopwords, contour_width=3, contour_color='firebrick')



wc.generate(text)

wc.to_file("wine.png")



plt.figure(figsize=[20, 10])

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()