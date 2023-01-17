
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import nltk
import re
import pandas as pd


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import matplotlib as mpl

import matplotlib.pyplot as plt
%matplotlib inline


doc1 = '../input/pmabiyahmed/abiy_ahmed_ali_nobel_lecture.txt'
with open(doc1, encoding='utf-8-sig') as file:
    df_nobel = file.read()
print (df_nobel[:400])
# convert all letters to lowercase in order to standardize the text.
df_nobel = df_nobel.lower()

# here, we just split the whole text by spaces 
tokens = [word for word in df_nobel.split()]
print(tokens[:100])
tokens = nltk.word_tokenize(df_nobel)
print(tokens[:100])
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(df_nobel)
print(tokens[:100])
token_freq = nltk.FreqDist(tokens)
stop_words = set(nltk.corpus.stopwords.words('english'))
# filter each sentence by removing the stop words
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
        
print(remove_stopwords(tokens)[:100])
token_freq = nltk.FreqDist(remove_stopwords(tokens))
token_freq.plot(25, cumulative=False)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=STOPWORDS,
                          max_words=300                          
                         ).generate(str(remove_stopwords(tokens)))

print(wordcloud)
fig = plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)
doc2 = '../input/pmabiyahmed/pm_abiy_inaugural_address.txt'
with open(doc2, encoding='utf-8-sig') as file:
    df_inaugural = file.read()
# convert all letters to lowercase in order to standardize the text.
df_inaugural = df_inaugural.lower()
tokens_inaug = tokenizer.tokenize(df_inaugural)

inaug_text = remove_stopwords(tokens_inaug)
token_freq = nltk.FreqDist(inaug_text)
token_freq.plot(25, cumulative=False)
wordcloud2 = WordCloud(
                          background_color='white',
                          stopwords=STOPWORDS,
                          max_words=300                          
                         ).generate(str(inaug_text))

print(wordcloud2)
fig = plt.figure(figsize=(20,10))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()


