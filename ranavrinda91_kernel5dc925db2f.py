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

        dataset =os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

movies_=pd.read_csv(dataset)

movies_.tail()

#Here we wil try to find out the genre of the movie from the plot of the movie
desc_=movies_[['Title','Plot','Release Year','Origin/Ethnicity']]

is_indian=desc_['Origin/Ethnicity']=="Bollywood" 

indian =desc_[is_indian]

indian.shape

indian.tail()
# choose some words to be stemmed

#stemming cuts off the end of the word or the beginning of the word.

#lemmatization takes into consideration into morphological analysis of the words.

#below is the demonstration



from nltk.stem import WordNetLemmatizer 

from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer() 

ps = PorterStemmer()   

words= ["finally","called","tries",'marriage','siblings','seduce','escape']

for w in words:

    print(w,ps.stem(w),"<<< by stemming")

    print(w, lemmatizer.lemmatize(w)," <<< by lemmatizing")

    #for experiment try to fist stem and then lemmatize

    stemmed=ps.stem(w)

    print(stemmed,lemmatizer.lemmatize(w)," <<< lemmatizing after stemming")
import nltk

import re

from nltk.corpus import stopwords

import string

from nltk.tokenize import sent_tokenize, word_tokenize   

from nltk.tokenize import WordPunctTokenizer

import gensim

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *



punc = WordPunctTokenizer()

lemmatizer = WordNetLemmatizer() 

stop = stopwords.words('english')

exclude = set(string.punctuation)

extra_stop=['the','he','she','they','charlie','jekyll','jack','film','tom','steve','andrews','jeff',

            'miranda','jonathan','alicia','john','one','jim','alice','henriette','richard','sylvia','andrew'

           ,'molly','pollyanna','mrs"','mrs','mr','louise','keaton','angela','mary','robert','paul','ann"','joe','bob',

           'however','sally','judy','jerry','jimmy','also','harold','david','marcus','margaret','nicki','harry',

           'brice','geoffrey','betty','morgan','juan','kelly','sebastian','cesar','maria','karl','egan','finds'

           ,'norma','another','tells','two','dorothy','vivian','billie','kitty','vivian','dan','young','man',

            'christine','eddie','nancy','davidson','david','ann','sadie','ronald','alvin','patricia','kiki','girl',

           'woman','takes','take','tells','tell','get','gets','have','stan','later','men','ivan','nick','anjali',

           'raja','killed','sonia','viktor','chris','ben','krishna','michael','adam','johnny','larry','duke','new',

           'mike','pete','elmer','bill','george','sam','susan','raj','sonali','anna','julia','oleg','joseph',

           'sergei','raju','kumar','vicky','henry','tony','boby','bobby','lily','raju','salim','kills',

           'sonia','oleg','julia','conan','peter','kiran','maya','james','singh','olga','philip','shiva',

           'singh','anton','abhi','arjun','alex','eric','billy','simon','rama','find','jackie','tina',

           'chandu','next','hari','kate','turn','first','leave','make','fall','soon','tries','try','final',

           'return','back','meet','want','come','here','call','called','leave','arrive','reach','away','far',

            'raja','sanjana','amar','vinay','vijay','rahul','rohit','shyam','prem','anand','aarti','ravi','priya',

           'radha','kiran','karan','vikram','suraj','ajay','ask','turn']

#here i have included names as stopwords becuase these names do not contirbute anything,

#towards finding the genre/topic of a movie

stop.extend(extra_stop)

stop=set(stop)

from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer() 

ps = PorterStemmer()



def clean_text(text):

    word_tokens = (word_tokenize(text))

    remove_stop=[w.lower() for w in word_tokens if w.lower() not in stop]

    remove_punct=[c for c in remove_stop if c not in exclude and len(c)>3]

    clean =[re.sub(r'[^a-zA-Z0-9]','',i) for i in remove_punct ]

    stemmer=[ps.stem(words) for words in clean]

    lemma= " ".join([lemmatizer.lemmatize(wr) for wr in stemmer])

    print (clean,stemmer,lemma)

    return lemma



indian['clean_plot'] = indian['Plot'].map(clean_text)

# create a word_freq dict



words_arr =indian['clean_plot'].values

word_freq={}

for sent in words_arr:

    for wr in word_tokenize(sent):

        if wr not in word_freq.keys():

            word_freq[wr]=1

        else:

            word_freq[wr]=word_freq[wr]+1

print (len(word_freq))            

lists = sorted(word_freq.items())

pd.DataFrame(lists,columns=['words','frequency']).sort_values(by='frequency',ascending=False).set_index('words')[:20].plot(kind='bar',figsize=(20,10),title='Frequency Dist For Top 20 Words');
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

import matplotlib.pyplot as plt



text = " ".join(text_ for text_ in indian['clean_plot'])

stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#Lets first build a Bag of Words model

processed_docs = [s.split(' ') for s in indian['clean_plot'].tolist()]

dictionary = gensim.corpora.Dictionary(processed_docs)

dictionary.filter_extremes(no_below=50, no_above=0.3)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

#We will build a tfidf corpus from the Bag of Words Model

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]
#Lets generate topics from the obtained tfidf corpus

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model_tfidf.print_topics(-1):

    print('Topic: {} Word: {}'.format(idx, topic))

    
from gensim.models import ldamodel

import gensim.corpora;

import pickle



array_text =[v for v in tokens_list_doc2bow]

id2word=gensim.corpora.Dictionary(array_text)

corpus = [id2word.doc2bow(text) for text in array_text]

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, 

                        num_topics=10)

lda.save('model5.gensim')

topics = lda.print_topics(num_words=5)

for topic in topics:

    print(topic)

print('\nPerplexity: ', lda.log_perplexity(corpus))  

# a measure of how good the model is. the lower the better.
