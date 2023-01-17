# text mining,knowledge discovery and question answering system
# CORD-19
# data source https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/data

# sections
# data cleaning and preprocessing
# EDA
# Clustering
# Cos Sim based Question Answering system

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
# Import the required libraries
import os
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import edit_distance

import re

import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
#df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
df.head() #title column, drop duplicates, drop na. rm stopwords, stem, rm punctuation. wordcloud
df.shape
df.isnull().sum()
df.dtypes
#show duplicates #some duplicates rmed after dataset updates
df.loc[df["title"]=="Risk Attitudes Affect Livestock Biosecurity Decisions With Ramifications for Disease Control in a Simulated Production System",]
    
    #show duplicates
df.loc[df["title"]=="Evaluation of scrub typhus diagnosis in China: analysis of nationwide surveillance data from 2006 to 2016",]
df1 = df[['title','abstract']]
df2=df1.drop_duplicates(keep='last')
    
    #show already dropped duplicates
df2.loc[df2["title"]=="Evaluation of scrub typhus diagnosis in China: analysis of nationwide surveillance data from 2006 to 2016",]
df2.isnull().sum()
df2.title.head(5)
df2.title.shape
df2.title.dropna(inplace=True)
df2.title.shape
# run too long, aborted
# string = ""
# for i,j in df.iterrows():
#     string += df2["title"]
#test with 100 rows
df2.head()
df3=df2.iloc[0:100,]
df3
out = ' '.join(df3["title"])
print (out)
# outString = ' '.join(df2["title"])
# print (outString)

# error before drop na



# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-68-e4c928dcad48> in <module>
# ----> 1 outString = ' '.join(df2["title"])
#       2 print (outString)

# TypeError: sequence item 19527: expected str instance, float found
# df2.iloc[19527,]



# Out[69]:
# title                  NaN
# abstract               NaN
# publish_time    2009 Sep 9
# journal          PLoS Curr
# Name: 23732, dtype: object
df2.title.iloc[19527,] #after drop na
df2.iloc[19527,]
outString = ' '.join(df2["title"])
print (outString)
text_file = open("outStringT.txt", "wt")
n = text_file.write(outString)
text_file.close()
# Function stopwords removal

def stopwords_removal(words) :
    stop_word = set(stopwords.words('english'))
    word_token = word_tokenize(words)
    output_sentence = [words for word in word_token if not word in stop_word]
    output_sentence = []
    for w in word_token:
        if w not in stop_word:
            output_sentence.append(w)
    return(output_sentence)
# call the function
stopwords_output = stopwords_removal(outString)
for w in stopwords_output:
    print(w+"|",end=' ')
type(stopwords_output)
#Function to Stem Words Using NLTK Library

def stems(words, method) :
    prtr = nltk.stem.PorterStemmer()
    snob = nltk.stem.SnowballStemmer('english')
    lema = nltk.wordnet.WordNetLemmatizer()
    
    word_to_stem = stopwords_removal(words)

    stem = [w for w in word_to_stem]
    stem = []
    
    if method == 'porter' :
        for w in word_to_stem:
            stem.append(prtr.stem(w))
 
    elif method == 'snowball': 
        for w in word_to_stem:
            stem.append(snob.stem(w))

    return (stem)
snowball_stems = stems(outString, "snowball")
print("After stemming, there are",len(snowball_stems),"words. And they are as following:")
print()
for s in snowball_stems:
    print(s+"|",end=' ')

# After stemming, there are 473262 words. And they are as following:
# After stemming, there are 444156 words. And they are as following:

### After stemming, there are 273901 words. And they are as following:
type(snowball_stems)
import string
x=snowball_stems
x = [''.join(c for c in s if c not in string.punctuation) for s in x]
x
#https://stackoverflow.com/questions/4371231/removing-punctuation-from-python-list-items
x = [s for s in x if s]
x
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 300
barWidth = 0.25
plt.figure(figsize=(20,15))

counts = Counter(x)
common = counts.most_common(50)
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)

plt.bar(np.arange(nbars), number,width=barWidth, tick_label=labels)
plt.xticks(rotation = 90, fontweight='bold',fontsize=12,)
plt.show()
barWidth = 0.25
# plt.figure(figsize=(20,15))

# counts = Counter(x)
# common = counts.most_common(100)
# labels = [item[0] for item in common]
# number = [item[1] for item in common]
# nbars = len(common)

# plt.bar(np.arange(nbars), number, width=barWidth,tick_label=labels) #
# plt.xticks(rotation = 90, fontweight='bold',fontsize=14,)
# plt.title('Top 100 words in titles of 24824 research papers',fontsize=15,fontweight='bold')
# plt.show()
df2.title.shape

plt.figure(figsize=(20,40))

counts = Counter(x)
common = counts.most_common(200)
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)

plt.barh(np.arange(nbars), number,tick_label=labels) #width=barWidth, 
plt.xticks( fontweight='bold',fontsize=12,) #rotation = 90,
plt.title('Top 200 words in titles of ' +str(df2.title.shape[0])+  ' research papers',fontsize=15)#,fontweight='bold'   #df2.title.shape
plt.show()

#how to make it in descending order
# want to rm custom stop words from common list
# common = counts.most_common(200).remove('in')
# for elem in common:
#     if elem % 3 == 'in':
#         common.remove(elem)
# try to creat wordcloud, Start with one title:
text = df2.title[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
text
#STOPWORDS
#type(STOPWORDS)

# not a exhausted list
# my observation from the previous output plots
# all research papers have method, abstract sections.
# e.g. "SARS COV","found" is high freq but low informative for drug discovery purpose
customize_stop_words2 = [
    'used', 'using', 'SARS CoV','MERS CoV','Abstract','found','result','method','conclusion','compared','many','well','including','identified','Although','present','Middle East','infection',
    'infectious','treatment','China','East','Role','COVID','human','model','Chapter','viruses'
]
#capital letter must match

STOPWORDS2 = list(STOPWORDS)  + customize_stop_words2
#STOPWORDS2 = list(STOPWORDS)  + customize_stop_words2
#stopwordsPlot = set(STOPWORDS2)  #able to run if it is a list not not a set
#stopwordsPlot = set(STOPWORDS2)

#snowball_stems
text = outString

# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords = STOPWORDS2,  background_color="white").generate(text)

#matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 300

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Top words cloud in titles of ' +str(df2.title.shape[0])+  ' research papers',fontsize=12)#,fontweight='bold'
plt.show()

# start work on abstract
df2.abstract.head(5)
df2.abstract.shape
df2.abstract.dropna(inplace=True)
df2.abstract.shape
outString_abstract = ' '.join(df2["abstract"])
print (outString_abstract)
# type(outString_abstract)
text_file = open("outStringA.txt", "wt")
n = text_file.write(outString_abstract)
text_file.close()
snowball_stems_abstract = stems(outString_abstract , "snowball")
print("After stemming, there are",len(snowball_stems_abstract),"words. And they are as following:")
print()
for s in snowball_stems_abstract:
    print(s+"|",end=' ')
    
    # run pretty long without GPU
    #After stemming, there are 6,181,753 words. And they are as following:

    #After stemming, there are 5816385 words. And they are as following:
    
    #After stemming, there are 3,849,953 words. And they are as following:
    #run about 10 mins on my computer
x2=snowball_stems_abstract
x2 = [''.join(c for c in s if c not in string.punctuation) for s in x2] #also took a whitle to rm punctuations

x2 = [s for s in x2 if s]
x2
df2.abstract.shape
plt.figure(figsize=(20,40))

counts = Counter(x)
common = counts.most_common(200)
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)

plt.barh(np.arange(nbars), number,tick_label=labels) #width=barWidth, 
plt.xticks( fontweight='bold',fontsize=12,) #rotation = 90,
plt.title('Top 200 words in abstracts of ' +str(df2.abstract.shape[0])+  ' research papers',fontsize=15)#,fontweight='bold'
#plt.title('Top 200 words in abstracts of '+ df2.abstract.shape+ ' research papers',fontsize=15)#,fontweight='bold'
plt.show()
#snowball_stems
text = outString_abstract

# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords = STOPWORDS2,background_color="white").generate(text) #background_color="white"

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 500

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Top words cloud in abstracts of  ' +str(df2.abstract.shape[0])+  ' research papers',fontsize=12)#,fontweight='bold'
plt.show()
#in addition to viz, show most common 1000 words since many words in top 100-300 may not be informative despite being frequently used
# i mannually 
common = counts.most_common(1000)
common 
type(common)
common
dfCommon = pd.DataFrame(common)
dfCommon.to_csv("./topwords1000.csv", sep=',',index=False)
