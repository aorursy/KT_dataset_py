# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pickle

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from textblob import TextBlob

from collections import Counter

from nltk.util import ngrams



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD 



#nltk.download('stopwords')

#nltk.download('punkt')





import os

#raw_data = pd.read_csv('/kaggle/input/alsa/pubmed_als_articles.csv')



# we are going to read in the data and parse the date column as dates

raw_data = pd.read_csv('/kaggle/input/alsa/pubmed_als_articles.csv', encoding='utf8', parse_dates=['publication_date'],index_col=None)



raw_data.head()
raw_data.iloc[0,:]
raw_data.iloc[0,0]

abstracts = raw_data['abstract']
# number of transcripts you want to analyze

fileids = range(0,51)

doc_sents = [TextBlob(abstracts[fileid])

             .sentences for fileid in fileids]



# let's look at a few

print(doc_sents[0][0:5])
doc_sents_nltk = [nltk.sent_tokenize(abstracts[fileid]) for fileid in fileids]



# to print the first one

print('\n-----\n'.join(nltk.sent_tokenize(abstracts[0])))
doc_words_word_tok = [nltk.word_tokenize(abstracts[fileid]) \

             for fileid in fileids]



print('\n-----\n'.join(nltk.word_tokenize(abstracts[0][123:170])))
doc_words_punct= [nltk.wordpunct_tokenize(abstracts[fileid]) \

             for fileid in fileids]



#to view a few

print('\n-----\n'.join(nltk.wordpunct_tokenize(abstracts[0][123:170])))
lemmizer = nltk.WordNetLemmatizer()



# note that i used the results from the word_punct tokenizer, but you can use any word tokenizer method



for w in doc_words_punct[0][31:55]:

        print(lemmizer.lemmatize(w), w)
stemmer = nltk.stem.porter.PorterStemmer()



for w in doc_words_punct[0][31:55]:

        print(stemmer.stem(w), w)
print(abstracts[19])
def _removeNonAscii(s): 

        return "".join(i for i in s if ord(i)<128)



_removeNonAscii(abstracts[19])



# for more than one doc     

#text_non_ascii = abstracts.map(lambda x: _removeNonAscii(x))
for w in doc_words_punct[0][33:37]:

    print(w.lower(), w)
for fileid in fileids[0:1]: 

    for w in doc_words_punct[0][31:55]:

        print(stemmer.stem(w.lower()),w)
from nltk.corpus import stopwords

stop = stopwords.words('english')

stop += ['.'," \'", 'ok','okay','yeah','ya','stuff','?']



clean_words = []

for doc in doc_words_punct:

    for w in doc:

        if w.lower() not in stop:

            clean_words.append(w.lower())

print(clean_words[0:10])
unclean_words = []

for w in doc_words_punct[0]:

    unclean_words.append(w.lower())

print(unclean_words[0:10])
def clean_text(text):

    

    """ 

    Takes in a corpus of documents and cleans. 

    

    1. remove any special strings and non ascii characters

    2. tokenize into words 

    3. lowercase and remove stop words

    4. lemmatize and  remove stop words again

    5. append to a list

    

    

    OUT: cleaned text = a list (documents) of lists (cleaned word in each doc)

        """

    

    # choose tokenizer

    tokenizer=nltk.wordpunct_tokenize

    

    # choose your stemmer

    #stemmer = WordNetLemmatizer().lemmatize

    stemmer = nltk.porter.PorterStemmer().stem

    #SnowballStemmer("english")

    

    

    #function to remove non-ascii characters

    def _removeNonAscii(s): 

        return "".join(i for i in s if ord(i)<128)







    stop_w = stopwords.words('english')

    stop_w += ['.', ',',':','...','!"','?"', " ' ","' "," '", '"'," - ","-"," — ",',"','."','!', ';',"/",\

             '.\'"','[',']',"—",".\'",'#','1','2','3','4','5','6','7','8','9',\

             ' oh ','la','was','wa','?','like' ," ' ",'I'," ? ","s", " t ","ve","re",\

              "(",")",").","'m","'s","\ 's","’","’","’","‘","n't","..."]

    stop_w = set(stop_w)



    cleaned_text = []

        

    text_non_ascii = text.map(lambda x: _removeNonAscii(x))

    

    for doc in text_non_ascii:

        cleaned_words = []

        

        for word  in tokenizer(doc):  

            low_word = word.lower()

            

            # throw out any words in stop words

            if low_word not in stop_w:

            

                # get roots

                root_word = stemmer(low_word)  

                          

                # keep if root is not in stopwords (yes, again)

                if root_word not in stop_w: 

                    

                # put into a list of words for each document, yes i lowered again, b/c i have trust issues

                    cleaned_words.append(root_word.lower())

        

        # keep corpus of cleaned words for each document    

        cleaned_text.append(' '.join(cleaned_words))

    

    return cleaned_text
clean_abs = clean_text(abstracts)
clean_abs[19]
# to save our cleaned text if using your local computer, not kaggle

#with open('cleaned_als_pubmed_abs.pkl', 'wb') as picklefile:

#    pickle.dump(cleaned_text, picklefile)
talks_blob = [TextBlob(abstracts[fileid]) for fileid in fileids]



# pulls all the nouns and all the things that are associated with it 

print('\n-----\n'.join(talks_blob[0][0:500].noun_phrases))
def n_grams_skr(data, n = 3, max_words=500):

    """extract ngrams and their counts, save into a dataframe. 

    INPUT: data = corpus, n=number of words to put in gram, max_words=number of most common ngrams to return

    OUTPUT: dataframe of most common ngrams and their counts in the corpus."""

    

    counter = Counter()

    df = pd.DataFrame(columns=['n_gram','count'])

    row = 0

    

    for doc in data:

        words = TextBlob(doc).words

        bigrams = ngrams(words, n)

        counter += Counter(bigrams)



    for phrase, count in counter.most_common(max_words):

        df.loc[row,'n_gram'] = phrase

        df.loc[row,'count'] = count

        row += 1

        

    return df
# this takes about 30 min to run with half the data...and you dont' need the output, 

# its just to get an idea of what our n-grams are

df = n_grams_skr(clean_abs[0:100],2,500)

df = df.sort_values('count',ascending=False)



df1 = n_grams_skr(clean_abs[0:100],3,500)

df1 = df1.sort_values('count',ascending=False)



dfb = pd.concat([df,df1], axis=1)

dfb.columns=['bi-gram','bi count','trigram','tri count']

dfb.head(20)
# CountVectorizer is a class; so `vectorizer` below represents an instance of that object.

c_vectorizer = CountVectorizer(ngram_range=(1,3), 

                             stop_words='english', 

                             max_df = 0.6, 

                             max_features=10000)



# call `fit` to build the vocabulary

c_vectorizer.fit(clean_abs)

# finally, call `transform` to convert text to a bag of words

c_data = c_vectorizer.transform(clean_abs)
%env JOBLIB_TEMP_FOLDER=/tmp
def topic_mod_als(vectorizer, vect_data, topics=20, iters=5, no_top_words=50):

    

    """ use Latent Dirichlet Allocation to get topics"""



    mod = LatentDirichletAllocation(n_components=topics,

                                    max_iter=iters,

                                    random_state=42,

                                    learning_method='online',

                                    n_jobs=-1)

    

    mod_dat = mod.fit_transform(vect_data)

    

    

    # to display a list of topic words and their scores 

    #next step is to make this into a matrix with all 5k terms and their scores from the viz?  

    

    def display_topics(model, feature_names, no_top_words):

        for ix, topic in enumerate(model.components_):

            print("Topic ", ix)

            print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-no_top_words - 1:-1]]) + '\n')

    

    display_topics(mod, vectorizer.get_feature_names() , no_top_words)



    

    return mod, mod_dat

mod, mod_dat = topic_mod_als(c_vectorizer, 

                            c_data, 

                            topics=20, 

                            iters=10, 

                            no_top_words=15)  
import pyLDAvis, pyLDAvis.sklearn

from IPython.display import display 

    

    # Setup to run in Jupyter notebooks

pyLDAvis.enable_notebook()



 # Create the visualization

vis = pyLDAvis.sklearn.prepare(mod, c_data, c_vectorizer,  sort_topics=False, mds='mmds')#, mds='tsne'



# Let's view it!

display(vis)
raw_data.index= raw_data['publication_date']



raw_data.abstract.resample('M').count().plot(style='-', figsize=(18,10))
raw_data.abstract.resample('A').count().plot(style='-', figsize=(18,10))
raw_data.resample('M').count().plot(style='-', figsize=(18,16))