# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from wordcloud import WordCloud

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

import nltk

import re

# Any results you write to the current directory are saved as output.
%matplotlib inline
def dataframe_structuring(season_no):

    print(str('../input/season')+str(season_no)+str('.json'))

    data_df =pd.read_json(str('../input/season')+str(season_no)+str('.json'))

    labels =data_df.columns

    feature=[]

    for item in labels:

        temp_df =data_df[item]

        temp_df =".".join(map(str,temp_df))

        feature.append(temp_df)

    return feature,labels
feature,label = dataframe_structuring(2)
def pre_processing(feature):

    temp_df = word_tokenize(feature)

    temp_df = nltk.pos_tag(temp_df)

    #dt_tags = [t for t in tags if t[1] == "DT"]

    df_tag=[]

    for t in temp_df:

        if t[1]=='NNP':

            if t[1]=='nan':

                continue

            else:

                df_tag.append(t[0])

    return df_tag


def wordcloud_cluster(feature):

    feature = " ".join(feature)

        

    noun_tag = pre_processing(feature)

    noun_tag = " ".join(noun_tag)

    noun_tag = re.sub(r'.nan+','',noun_tag)

    stop_words = set(stopwords.words("english"))

    wordcloud = WordCloud(width = 1600, height = 1200, 

                background_color ='black', 

                stopwords = stop_words, 

                min_font_size = 10).generate(noun_tag)

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.show()

    return plt
wordcloud_cluster(feature)
wordcloud_cluster(feature[1])
wordcloud_cluster(feature[2])
wordcloud_cluster(feature[3])