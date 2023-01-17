#Importing all required libraries 
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
#import the data 
df = pd.read_csv("../input/jio-mart-review/reviews_googleplay_1590637826.csv",skiprows=0,encoding= 'unicode_escape')
df
sorted_data = df[["Author","Review"]]
sorted_data
sorted_data.Review
#This are the review for thr jio mart(Relaince Digital shopping online platform) application 
df.Review

text = " ".join(review for review in df.Review)
print("There are {} words in the combination of all review.".format(len(text)))
#count of the NaN value float found means NaN value.
df['Review'].isnull().sum()
df['Review'].isnull().values.any()

#Replacing the NaN value by jio
df["Review"].fillna("jio", inplace = True) 
df['Review'].isnull().sum()
#Here we get zero number of NaN VALUE
text = " ".join(review for review in df.Review)
print(" There are {} words  in the combination of all review.".format(len(text)))
#Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["Gd", "Nyc"])
#Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords,background_color="white").generate(text)



#Display the generated image:
# The matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
