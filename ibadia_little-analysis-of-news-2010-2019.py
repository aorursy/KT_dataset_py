# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from wordcloud import WordCloud

from wordcloud import WordCloud, STOPWORDS



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/categories_data.csv")

data.head()
year_dictionary={}

stopwords = set(STOPWORDS)

stopwords=list(stopwords)+["in","to","on","off","for","us"]

stopwords=set(stopwords)

for i,x in data.iterrows():

    year=x['Date'][:4]

    if x['Date'][:4] not in year_dictionary:

        year_dictionary[x['Date'][:4]]={}

    word_list=x['Headline'].lower().strip().split(' ')

    for word in word_list:

        if word in stopwords:

            continue

        if word not in year_dictionary[year]:

            year_dictionary[year][word]=0

        year_dictionary[year][word]+=1



def generate_wordcloud(dictionary, year_name):

    wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=80, 

                          random_state=42

                         ).generate_from_frequencies(dictionary)

    

    fig = plt.figure(1)

    plt.imshow(wordcloud)

    plt.title(year_name)

    plt.axis('off')

    plt.show()

    fig.savefig(year_name+".png", dpi=1000)



    





    

for x in year_dictionary.keys():

    print ("Generating wordCloud for year: "+str(x))

    generate_wordcloud(year_dictionary[x], x)

print ("completed")