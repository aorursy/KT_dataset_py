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

ep_num = len([name for name in os.listdir('../input/friends-tv-series-screenplay-script')])





print("Friends Season consists of {} episodes.".format(ep_num))
import glob

texts = ""

folder_name = "../input/friends-tv-series-screenplay-script/"

for f in glob.glob(folder_name + '/*.txt'):

    temp = open(f,'r')

    texts += temp.read()

    temp.close()
len(texts)
text = re.sub('[^A-Za-z]+', ' ', texts)
# adding screenplay notes to stopwords

nlp = spacy.load("en")

nlp.Defaults.stop_words |= {"d","ll","m","re","s","ve", "t", "oh", "uh", "na", "okay",

                           "didn","don","gon","j","hm","um","dr","room","int", "ext", 

                           "cut", "day", "night", "theme", "tune","music", "ends","view","opening credits scene", 

                            "commercial break scene", "hey hey hey", "hey", "closing credits scene","scene",

                            "closeup", 'freshly', 'squeezed', 'fade'}

stopwords = nlp.Defaults.stop_words
# function to find and plot frequent words

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

plot_words(all_words_no_stop.most_common(10), "Top 10 frequent words")
bigram = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(bigram.most_common(10), "Top 10 frequent bigrams.")
trigrams = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(trigrams.most_common(10), "Top 10 frequent trigrams.", "#2b2e2b")
characters = [

'monica','rachel','ross','joey','chandler','phoebe','central perk',"opening credits scene", 

"commercial break scene", "hey hey hey", "hey", "closing credits scene","scene"]



# unique names only

names = set(" ".join(set(characters)).lower().split())



nlp.Defaults.stop_words |= names
no_names = nltk.FreqDist(w.lower() for w in all_words if w not in stopwords)

plot_words(no_names.most_common(10), "Top 10 frequent words except for names")
no_names_bigram = nltk.FreqDist(nltk.bigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(no_names_bigram.most_common(10), "Top 10 frequent bigrams except for names")
no_names_trigram = nltk.FreqDist(nltk.trigrams(w.lower() for w in all_words if w not in stopwords))

plot_ngrams(no_names_trigram.most_common(10), "Top 10 frequent trigrams except for names", "#2b2e2b") 
# the mask image taken from http://www.designcenterassoc.com/wp-content/uploads/2017/11/Friends-PNG-HD-e1509653607131.png

# cooper_mask = np.array(Image.open('../input/temporary/Friends-PNG-HD-e1509653607131.png'))



def color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(0, 100%, 27%)"



wc = WordCloud(background_color="white", max_words=1000,

               stopwords=stopwords, contour_width=4, contour_color='steelblue')



wc.generate(" ".join(all_words_no_stop.keys()))



plt.figure(figsize=(18, 10))

plt.imshow(wc.recolor(color_func=color_func, random_state=3),interpolation="bilinear")

plt.axis("off")
"Well, exactly {} times".format(all_words_no_stop['coffee'])
"It was mentioned {} times throughout all episodes".format(all_words_no_stop['doin'])