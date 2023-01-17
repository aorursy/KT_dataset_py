from wordcloud import WordCloud,STOPWORDS

from nltk.corpus import stopwords

from nltk.corpus import wordnet

from nltk import word_tokenize

from nltk.corpus import state_union

from nltk.tag import pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from collections import Counter

import matplotlib.pyplot as plt

import collections, re

import string

import pylab as plt

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df=pd.read_csv('../input/reddit_worldnews_start_to_2016-11-22.csv')

df1=df[df.up_votes>5000]

print(df1.head(5))
#stopwords removed and as sentence



filtered_words=df1['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))



bagsofwords_1 = [ collections.Counter(re.findall(r'\w+', txt)) for txt in filtered_words]

sumbags = sum(bagsofwords_1, collections.Counter())



#to find the proper nouns in the filtered words

A=pd.Series.to_string(filtered_words)

tagged_sent = pos_tag(A.split())

propernouns = [word for word,pos in tagged_sent if pos == 'NNP']



propernouns_count=Counter(propernouns)
#wordcloud of most common 50 proper nouns



most_common_50 = propernouns_count.most_common(51)

most_common_50 = pd.DataFrame(most_common_50)

most_common_50 = most_common_50.drop(most_common_50.index[1])

most_common_50 = most_common_50.drop(most_common_50.index[38])



most_common_50.rename(columns={0:'text',1:'score'},inplace=True)

wordcloud1 = WordCloud(background_color='white', width=3000, height=2500).generate(' '.join(most_common_50['text']))



plt.figure(1,figsize=(8,8))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
sid=SentimentIntensityAnalyzer()

def senti_analysis(wordlist):

    global local_vars

    wordlist_1=[wordlist]

    bag_of_sentences = [sentence for sentence in filtered_words if any(word in sentence for word in wordlist_1)]

    ss=[]

    senti_vals=[]    

    for sentence in bag_of_sentences:   

        ss.append(sid.polarity_scores(sentence))

    senti_vals = [i['compound'] for i in ss]

    senti_val =  sum(senti_vals)/len(senti_vals)

    senti_val_positive = filter(lambda ss: ss['compound']>0,ss)

    senti_val_negative = filter(lambda ss: ss['compound']<0,ss)

    return senti_val



most_common_50_1= []

most_common_50['text'] =most_common_50['text'].astype(str)



i=0           

for word in most_common_50['text']:

    most_common_50_1.append(senti_analysis(word))

    i=i+1

    

most_common_50['senti_val'] = pd.Series(most_common_50_1, index=most_common_50.index)



X = np.arange(0,len(most_common_50))

y=most_common_50['senti_val']



LABELS = most_common_50['text']



ax=plt.bar(X,y, align='center', width=0.5)

#ax.autoscale(tight=True)

plt.xticks(X, LABELS, rotation='vertical',fontsize=7)

plt.show()
