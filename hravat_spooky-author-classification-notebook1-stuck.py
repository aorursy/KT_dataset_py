##import necessary packages

import numpy as np # linear algebra

import scipy as sp 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns 

import nltk

from nltk.corpus import stopwords

import wordcloud

from wordcloud import WordCloud, STOPWORDS

from matplotlib import pyplot as plt

##read in the data file an display 

df_spooky_author=pd.read_csv('../input/train.csv')

df_spooky_author



sentence_list=df_spooky_author['text'].values.tolist()

author_list=df_spooky_author['author'].values.tolist()



sentence_list

author_list





combined_list=[list(author_list) for author_list in zip(author_list, sentence_list)]



tokenized_list=[]



for authors,sentence in combined_list:

    tokenized_list.append([authors,nltk.word_tokenize(sentence)])



tokenized_list



##make individual authorwise list of words after removing stop words.

##This is to prepare the data for wordcloud

stop = set(stopwords.words('english'))    



tokenized_stop_words_list=[]





for author,sentence in tokenized_list:

    tokenized_stop_words_list_temp=[]   

    for word in sentence:

        if not word in stop:

            tokenized_stop_words_list_temp.append(word)

    tokenized_stop_words_list.append([author,tokenized_stop_words_list_temp])        

            

tokenized_stop_words_list



tokenized_words_EAP=[]

tokenized_words_MWS=[]

tokenized_words_HPL=[]



for author,sentence in tokenized_stop_words_list:

    if author=='EAP':

        for words in sentence:

            tokenized_words_EAP.append(words)

    if author=='MWS':

        for words in sentence:

            tokenized_words_MWS.append(words)

    if author=='HPL':

        for words in sentence:

            tokenized_words_HPL.append(words)

            

##word cloud for HPLovenCraft            

plt.figure(figsize=(30,30))

wc = WordCloud(background_color="black", max_words=10000, 

               stopwords=stop, max_font_size= 40)

wc.generate(" ".join(tokenized_words_HPL))

##plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=16)

plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.axis('off')
