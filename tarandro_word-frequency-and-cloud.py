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
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.head(20)
train_df['target'].plot(kind='hist')
# Import python packages

import os

from os import walk

import shutil

from shutil import copytree, ignore_patterns

from PIL import Image

from wand.image import Image as Img

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



# from https://www.kaggle.com/benhamner/most-common-forum-topic-words :



topic_words = [ z.lower() for y in

                   [ x.split() for x in train_df[train_df["target"] == 1]['text'] if isinstance(x, str)]

                   for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]



# choose only significant words in /popular_words_nonstop[0:30]/:

popular_target1 = ['fire','via','suicide','disaster','police','people','killed','like','california','families',

 'two','storm','train','2','bomb','emergency','get','crash','one','nuclear','bombing','news','fires','northern',

                   'buildings','burning','hiroshima']



plt.barh(range(27), [word_count_dict[w] for w in reversed(popular_target1)])

plt.yticks([x + 0.5 for x in range(27)], reversed(popular_target1))

plt.title("Popular Words in target = 1")

plt.show()
plt.figure(figsize=(15,15))

# from https://www.kaggle.com/paultimothymooney/explore-job-postings :

topic_words = [ z.lower() for y in

                       [ x.split() for x in train_df[train_df["target"] == 1]['text'] if isinstance(x, str)]

                       for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

word_string=str(popular_words_nonstop[0:50])



# choose only significant words in word_string:

popular_target1= "[fire,via,suicide,disaster,police,people,killed,like,california,families,two,storm,train,2,bomb,emergency,get,crash,one,nuclear,bombing,news,fires,northern,buildings,burning,hiroshima,bomber,dead,still,atomic,war,homes,fatal,'#news',new,obama,car,years,accident,debris,may,attack,wildfire,first,watch]"



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=50,

                          width=1000,height=1000,

                         ).generate(popular_target1)

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
topic_words = [ z.lower() for y in

                   [ x.split() for x in train_df[train_df["target"] == 0]['text'] if isinstance(x, str)]

                   for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]



# choose only significant words in /popular_words_nonstop[0:30]/:

popular_target0 = ['like',"i'm",'??','new','get','one','body','via','would','love','got','people','full','see',

 'know','2','video','back','emergency','????','going','still','time',"can't",'want','@youtube','go']



plt.barh(range(27), [word_count_dict[w] for w in reversed(popular_target0)])

plt.yticks([x + 0.5 for x in range(27)], reversed(popular_target0))

plt.title("Popular Words in target = 0")

plt.show()
plt.figure(figsize=(15,15))

# adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

topic_words = [ z.lower() for y in

                       [ x.split() for x in train_df[train_df["target"] == 0]['text'] if isinstance(x, str)]

                       for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

word_string=str(popular_words_nonstop[0:50])



# choose only significant words in word_string:

popular_target0= """[like,i'm,??,new,get,one,body,via,would,love,got,people,full,see,know,2,video,back,emergency,????,going,still,time,can't,want,@youtube,go,think,us,day,fire,good,u,last,first,3,make,need,man,world,really,many,even,burning,let,take,way]"""



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=50,

                          width=1000,height=1000,

                         ).generate(popular_target0)

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
test_df.head(10)
topic_words = [ z.lower() for y in

                   [ x.split() for x in test_df['text'] if isinstance(x, str)]

                   for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]



# choose only significant words in /popular_words_nonstop[0:30]/:

popular = ['like','get','via',"i'm",'new','fire','??','one','would','people','2','emergency','attack','suicide',

 'first','...','police','still','rt','full','disaster','video','got','news','going','two','storm']



plt.barh(range(27), [word_count_dict[w] for w in reversed(popular)])

plt.yticks([x + 0.5 for x in range(27)], reversed(popular))

plt.title("Popular Words in test set")

plt.show()
plt.figure(figsize=(15,15))

# adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

topic_words = [ z.lower() for y in

                       [ x.split() for x in test_df['text'] if isinstance(x, str)]

                       for z in y]

word_count_dict = dict(Counter(topic_words))

popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

word_string=str(popular_words_nonstop[0:50])



# choose only significant words in word_string:

popular= "[like,get,via,i'm,new,fire,??,one,would,people,2,emergency,attack,suicide,first,...,police,still,rt,full,disaster,video,got,news,going,two,storm,burning,last,3,watch,????,love,see,fires,make,body,us,know,think,time,need,even,go,car,back,never]"



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=50,

                          width=1000,height=1000,

                         ).generate(popular)

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()