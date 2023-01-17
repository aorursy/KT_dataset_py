# Standard data processing packages: numpy and pandas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Text cleaning and string formating tool box: nltk, string and re. In particular, stopwords and word_tokenize are great tools for turning text 

# into tokens and remove common words; Wordnetlemmatizer allows reasonable lemmatization; re and string use for removing punctuations and other 

# string wrangling

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

import re

import string

from langdetect import detect



#CountVectorizer() for counting word tokens and creating document term matrix needed for Gensim. Also require the text package to modify 

#stopwords according to what we see in the EDA



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import text



#Gensim for topic modelling. Requires scipy.sparse to store spare matrix, and use Coherence Model to evalute topics

from gensim import matutils, models

import scipy.sparse

from gensim.models import CoherenceModel

from gensim import corpora



#Visualisation tools. pyLDAvis to visualise the topics, matplotlib for general plotting

import pyLDAvis

import pyLDAvis.gensim  

import matplotlib.pyplot as plt



#Pickle for saving model

import pickle



#Load in data

meta_df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

meta_df.head()

#See how big the dataset is

print("size of table is ", meta_df.shape)
#Create new dataframe that just store the title and abstract to work on

text_df = meta_df[['sha','title','abstract']].copy()



#Print out title to see what we are in for

print(text_df.title[345])



#define a text cleaner to do the cleaning we need - lower case, and replace all punctuation with empty strings

#Note the explict casting of the title elements into str is needed to use string operations without an error

def text_cleaner(text):

    text = str(text).lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)



    return text





text_df.title = text_df.title.apply(lambda x: text_cleaner(x))

text_df = text_df[text_df['title'].notna()]

text_df.head()
#Use CountVectorizer to vectorize the counts of each word in the title column. i.e. creating a document term matrix

cv = CountVectorizer(stop_words = "english")

text_cv = cv.fit_transform(text_df.title)

dtm = pd.DataFrame(text_cv.toarray(), columns = cv.get_feature_names())

dtm.head()

#Identify top words by aggregating the table

top_words = dtm.sum(axis = 0).sort_values(ascending = False)

print(top_words[0:50])
#combine the different "virus" forms and combine the term "severe acute respiratory syndrome" into "sars"

text_df.title = text_df.title.apply(lambda x:x.replace("severe acute respiratory syndrome", "sars"))

text_df.title = text_df.title.apply(lambda x:re.sub('viral|viruses', 'virus', x))



#Remove pure numbers from the text

text_df.title = text_df.title.apply(lambda x:re.sub('[0-9]', '', x))





#Lemmatization for the rest of the words using wordnet lemmatizer from nltk. A new column "Tokens" is formed in the dataframe to store this

wordnet_lemmatizer = WordNetLemmatizer()

lemma = WordNetLemmatizer()

text_df['Tokens'] = text_df.title.apply(lambda x: word_tokenize(x))

text_df.Tokens = text_df.Tokens.apply(lambda x: " ".join([lemma.lemmatize(item) for item in x]))





#Add stop_words of "chapter", "study", "virus" and redo the countvectorizer. Stopwords can be manually formed using text.ENGLISH_STOP_WORDS

stop_words = text.ENGLISH_STOP_WORDS.union(["chapter","study","virus"])

cv2 = CountVectorizer(stop_words = stop_words)

text_cv_stemed = cv2.fit_transform(text_df.Tokens)

dtm = pd.DataFrame(text_cv_stemed.toarray(), columns = cv2.get_feature_names())

top_words = dtm.sum(axis = 0).sort_values(ascending = False)

print(top_words[0:50])



#First, the dtm needs to be transposed into a term document matrix, and then into a spare matrix

tdm = dtm.transpose()

sparse_counts = scipy.sparse.csr_matrix(tdm)



#Gensim provide tools to turn the spare matrix into the corpus input needed for the LDA modelling. 

corpus = matutils.Sparse2Corpus(sparse_counts)



#One also require a look up table that allow us to refer back to the word from its word-id in the document term matrix.

id2word = dict((v,k) for k, v in cv2.vocabulary_.items())



#Fitting a LDA model simply requires the corpus input, the id2word look up, and specify the number of topics required

lda = models.LdaModel(corpus = corpus, id2word = id2word, num_topics=20, 

                                           passes=10,

                                           alpha='auto')



lda.print_topics()



#The Coherence model uses a corpora.Dictionary object that have both a word2id and id2word lookup table. We can create this dictionary as follows

d = corpora.Dictionary()

word2id = dict((k, v) for k, v in cv2.vocabulary_.items())

d.id2token = id2word

d.token2id = word2id



#Function to create LDA model and evalute the coherence score for a range of values for the number of topics. Note that the coherence model needs

#the original text to calculate the coherence, i.e. the tokens column in the table. The column needs to be tokenized as it was stored as strings

#in the dataframe.

def calculate_coherence(start, stop, step, corpus, text, id2word, dictionary):

    model_list = []

    coherence = []

    for num_topics in range(start, stop, step):

        lda = models.LdaModel(corpus = corpus, id2word = id2word, num_topics=num_topics, passes=10,alpha='auto', random_state = 1765)

        model_list.append(lda)

        coherence_model_lda = CoherenceModel(model=lda, texts=text, dictionary=dictionary, coherence='c_v')

        coherence_lda = coherence_model_lda.get_coherence()

        print("Coherence score for ", num_topics, " topics: ", coherence_lda)

        coherence.append(coherence_lda)

    

    return model_list, coherence



#Create and evaluate models with 10 - 80 topics in steps of 10

model_list, coherence_list = calculate_coherence(10, 90, 10, corpus, text_df.Tokens.apply(lambda x: word_tokenize(x)), id2word, d)





# Plot graph of coherence score as a function of number of topics to look at optimal number of topics. 

x = range(10, 90, 10)

plt.plot(x, coherence_list)

plt.xlabel("Number of Topics")

plt.ylabel("Coherence score")

plt.show()



model_list[6].show_topics(num_topics=20, num_words=10, log=False, formatted=True)



# Visualize the topics

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(model_list[6], corpus, d)

vis
#Store tht top 3 topics, the contribution of the most dominant topic, and the total contribution of the top three topics

topic_list = []

top_score = []

sum_score = []

#lda_mode[corpus] gives the list of topics for each sentence (list of tokens) in the corpus as a list of tuples. We can then decipher that 

#and extract out the top three topics and their scores

for i, row in enumerate(model_list[6][corpus]):

    top_topics = sorted(row, key=lambda tup: tup[1], reverse = True)[0:3]

    topic_list.append([tup[0] for tup in top_topics])

    top_score.append(top_topics[0][1])

    sum_score.append(sum([tup[1] for tup in top_topics]))



text_df['topics'] = topic_list

text_df['top_score'] = top_score

text_df['sum_scores'] = sum_score

text_df.head()
#Finally, save the models ,the topic keys, and the updated dataframe for later use

model_file = open("title_model",'wb')

pickle.dump(model_list[6], model_file)

model_file.close()



dict_file = open("dictionary",'wb')

pickle.dump(d, dict_file)

dict_file.close()



text_df.to_csv("topic_data.csv", index = False, header=True)
