# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

import matplotlib                  # 2D Plotting Library        

import geopandas as gpd            # Python Geospatial Data Library

plt.style.use('fivethirtyeight')

%matplotlib inline

#import nltk

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize,sent_tokenize



#preprocessing

from nltk.corpus import stopwords  #stopwords

from nltk import word_tokenize,sent_tokenize # tokenizing

from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet



# for named entity recognition (NER)

from nltk import ne_chunk



# vectorizers for creating the document-term-matrix (DTM)

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.decomposition import TruncatedSVD



#stop-words

stop_words=set(nltk.corpus.stopwords.words('english'))
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

df.head()
index = df.index

number_of_rows = len(index)

print(number_of_rows)
print (new1)


def clean_text(review):

    le=WordNetLemmatizer()

    word_tokens=word_tokenize(review)

    tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]

    cleaned_text=" ".join(tokens)

    return cleaned_text
new1['cleaned_text']=new1['review'].apply(clean_text)
new1.head()
new1.drop(['review'],axis=1,inplace=True)
vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)

vect_text=vect.fit_transform(new1['cleaned_text'])

print(vect.get_feature_names())
print(vect_text.shape)

type(vect_text)
idf=vect.idf_
dd=dict(zip(vect.get_feature_names(), idf))

l=sorted(dd, key=(dd).get)

# print(l)

print(l[0],l[-1])

print(dd['like'])

print(dd['success'])  
new1['cleaned_text'].head()
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)



lsa_top=lsa_model.fit_transform(vect_text)
print(lsa_top[0])

print(lsa_top.shape)  # (no_of_doc*no_of_topics)




l=lsa_top[0]

print("Document 0 :")

for i,topic in enumerate(l):

  print("Topic ",i," : ",topic*100)



print(lsa_model.components_.shape) # (no_of_topics*no_of_words)

print(lsa_model.components_)

# most important words for each topic

vocab = vect.get_feature_names()



for i, comp in enumerate(lsa_model.components_):

    vocab_comp = zip(vocab, comp)

    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]

    print("Topic "+str(i)+": ")

    for t in sorted_words:

        print(t[0],end=" ")

    print("\n")
from sklearn.decomposition import LatentDirichletAllocation

lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=1) 

# n_components is the number of topics




lda_top=lda_model.fit_transform(vect_text)



print(lda_top.shape)  # (no_of_doc,no_of_topics)

print(lda_top[0])
sum=0

for i in lda_top[0]:

  sum=sum+i

print(sum)

# composition of doc 0 for eg

print("Document 0: ")

for i,topic in enumerate(lda_top[0]):

  print("Topic ",i,": ",topic*100,"%")





print(lda_model.components_[0])

print(lda_model.components_.shape)  # (no_of_topics*no_of_words)



# most important words for each topic

vocab = vect.get_feature_names()



for i, comp in enumerate(lda_model.components_):

    vocab_comp = zip(vocab, comp)

    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:5]

    print("Topic "+str(i)+": ")

    for t in sorted_words:

        print(t[0],end=" ")

    print("\n")
from wordcloud import WordCloud

# Generate a word cloud image for given topic

def draw_word_cloud(index):

  imp_words_topic=""

  comp=lda_model.components_[index]

  vocab_comp = zip(vocab, comp)

  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:25]

  for word in sorted_words:

    imp_words_topic=imp_words_topic+" "+word[0]



  wordcloud = WordCloud(width=900, height=600).generate(imp_words_topic)

  plt.figure( figsize=(5,5))

  plt.imshow(wordcloud)

  plt.axis("off")

  plt.tight_layout()

  plt.show()
# topic 0

draw_word_cloud(0)