import re

import os

import nltk

import spacy

import random

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from PIL import Image

from wordcloud import WordCloud



from nltk import word_tokenize

from nltk.util import ngrams
# count the number of episodes in each season

s1_ep_num = len([name for name in os.listdir('/kaggle/input/scripts/season1/')])

s2_ep_num = len([name for name in os.listdir('/kaggle/input/scripts/season2/')])



print("Season 1 consists of {} episodes.".format(s1_ep_num))

print("Season 2 consists of {} episodes.".format(s2_ep_num))
texts = ""

for ep_name in range(1, s1_ep_num+1):

    with open(os.path.join("/kaggle/input/scripts/season1/", 

                           str(ep_name)+".txt")) as f:

        texts += f.read()
len(texts)
text = re.sub('[^A-Za-z]+', ' ', texts)
# adding screenplay notes to stopwords

nlp = spacy.load("en")

nlp.Defaults.stop_words |= {"d","ll","m","re","s","ve", "t", "oh", "uh", "na", "okay",

                           "didn","don","gon","j","hm","um","dr","room","int", "ext", 

                           "cut", "day", "night", "theme", "tune","music", "ends","view",

                            "closeup", 'freshly', 'squeezed', 'fade'}

stopwords = nlp.Defaults.stop_words
def plot_words(words,title,color="#114d1e"):

    counts = {}

    for i in range(len(words)):

        counts[words[i][0]] = words[i][1]

    plt.figure(figsize=(8,6))

    plt.title(title, fontsize=14)

    plt.barh(range(len(counts)), list(counts.values()), color=color, align="center")

    plt.yticks(range(len(counts)), list(counts.keys()), fontsize=12)

    plt.gca().invert_yaxis()

    plt.show()

    

def plot_ngrams(ngrams,title,color="#7a2822"):

    counts = {}

    for i in range(len(ngrams)):

        counts[" ".join(ngrams[i][0])] = ngrams[i][1]

    plt.figure(figsize=(8,6))

    plt.title(title, fontsize=14)

    plt.barh(range(len(counts)), list(counts.values()), color=color,align="center")

    plt.yticks(range(len(counts)), list(counts.keys()), fontsize=12)

    plt.gca().invert_yaxis()

    plt.show()
all_words = nltk.tokenize.word_tokenize(text.lower())

all_words_no_stop = nltk.FreqDist(w.lower() for w in all_words if w not in stopwords)

plot_words(all_words_no_stop.most_common(10), "Top 10 frequent words. Season 1")
bigram = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(bigram.most_common(10), "Top 10 frequent bigrams. Season 1")
trigrams = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(trigrams.most_common(10), "Top 10 frequent trigrams. Season 1", "#2b2e2b")
characters = [

'Dale Cooper',

'Sheriff Harry Truman',

'Shelly Johnson',

'Bobby Briggs',

'Benjamin Horne',

'Donna Hayward',

'Audrey Horne',

'Will Hayward',

'Norma Jennings',

'James Hurley',

'Ed Hurley',

'Pete Martell',

'Leland Palmer',

'Josie Packard',

'Catherine Martell',

'Lucy Moran',

'Laura Palmer',

'Lawrence Jacoby',

'Leo Johnson',

'Eileen Hayward',

'Andy Brennan',

'Mike Nelson',

'Tommy Hawk Hill'

'Sarah Palmer',

'Jacques Renault',

'Windom Earle',

'Ronette Pulaski',

'Phillip Jeffries',

'Albert Rosenfield',

'Teresa Banks',

'Annie Blackburn',

'Chester Desmond',

'Gordon Cole',

'Carl Rodd',

'Sam Stanley',

'Harold Smith'

]



# unique names only

names = set(" ".join(set(characters)).lower().split())



nlp.Defaults.stop_words |= names
no_names = nltk.FreqDist(w.lower() for w in all_words if w not in stopwords)

plot_words(no_names.most_common(10), "Top 10 frequent words except for names")
no_names_bigram = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(no_names_bigram.most_common(10), "Top 10 frequent bigrams except for names")
no_names_trigram = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(no_names_trigram.most_common(10), "Top 10 frequent trigrams except for names", "#2b2e2b") 
# the mask image taken from https://www.reddit.com/r/twinpeaks/comments/2mtbtf/that_other_post_made_me_remember_heres_a_picture/

cooper_mask = np.array(Image.open('/kaggle/input/masks/cooper_mask.png'))



def color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(0, 100%, 27%)"



wc = WordCloud(background_color="white", mask=cooper_mask, max_words=1000,

               stopwords=stopwords, contour_width=4, contour_color='steelblue')



wc.generate(" ".join(all_words_no_stop.keys()))



plt.figure(figsize=(18, 10))

plt.imshow(wc.recolor(color_func=color_func, random_state=3),interpolation="bilinear")

plt.axis("off")
"Well, exactly {} times".format(all_words_no_stop['coffee'])
"It was mentioned {} times throughout 8 episodes".format(all_words_no_stop['pie'])