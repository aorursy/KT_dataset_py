# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

from stop_words import get_stop_words

import re

import seaborn as sns

import random

from os import path



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
d = path.dirname(__file__)
debate = pd.read_csv('../input/debate.csv',encoding = 'iso-8859-1')

debate.head(10)
debate = debate.loc[debate.Date == "2016-09-26"]

debate.Speaker.drop_duplicates()
CLINTON = "Clinton"

TRUMP = "Trump"
debate[(debate.Speaker == "CANDIDATES") & (debate.Text != "(CROSSTALK)")]
debate[debate['Text'].str.contains("Wrong")]
debate[debate['Text'].str.contains("China")]
[x.count("China") for x in debate[debate['Text'].str.contains("China")].Text]
stopwords = set(get_stop_words('en'))

stopwords |= {"look", "thing", "say", "said", "will", "well", "also"}



def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 100%%, %d%%)" % random.randint(20, 60)



def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(240, 100%%, %d%%)" % random.randint(20, 60)



def getWordCloud(candidate, col):

    words = " ".join(debate.loc[debate.Speaker == candidate]["Text"])

    wordcloud = WordCloud(max_font_size=40, relative_scaling=0.5,

                          background_color="white",

                          stopwords=stopwords,random_state=1).generate(words)

    

    plt.figure(figsize=(21,28))

    plt.imshow(wordcloud.recolor(color_func=col, random_state=3))

    plt.title(candidate + "'s Words in Debate")

    plt.axis("off")

    #plt.show()

    return plt
getWordCloud(TRUMP, red_color_func)
getWordCloud(CLINTON, blue_color_func)
#Extracting words from a string

#removing punctuation and returning a list with separated words

def getWords(text):

    return re.compile('\w+').findall(text)
#using word count as a proxy for time.

debate["Length"] = debate.Text.map(getWords).map(len)
plt.figure(figsize=(14,8))

markerline, stemlines, baseline = plt.stem(debate[debate.Speaker == CLINTON].Line, 

               debate[debate.Speaker == CLINTON].Length,

               markerfmt=' ', label= CLINTON)

plt.setp(stemlines, 'color', 'b')

plt.setp(baseline, visible=False)



markerline, stemlines, baseline = plt.stem(debate[debate.Speaker == TRUMP].Line, 

               debate[debate.Speaker == TRUMP].Length,

               markerfmt=' ', label= TRUMP)

plt.setp(stemlines, 'color', 'r')

plt.setp(baseline, visible=False)

plt.legend()

plt.title("Candidates' Response Lengths over Time")

plt.show()