import numpy as np # linear algebra

import pandas as pd #for io

import gensim  #for topic modelling

from bs4 import BeautifulSoup #for extracting text 

import nltk # for text preprocessing

import re 
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

question_df = pd.read_csv("../input/Questions.csv",encoding='latin1')

tags_df = pd.read_csv("../input/Tags.csv",encoding='latin1')

question_df.head()
modified_question_df = question_df.drop(['Score','CreationDate','OwnerUserId','Id'],axis = 1)

number_of_questions = 800

titles = modified_question_df.Title[:number_of_questions]

questions = modified_question_df.Body[:number_of_questions]
import string



def display_visible_html_using_re(text):             

    return (re.sub("(\<.*?\>)", "",text))



def remove_punctuation(text):

        exclude = set(string.punctuation)

        return ''.join(ch for ch in text if ch not in exclude)



questions = [display_visible_html_using_re(question) for question in questions]

titles = [remove_punctuation(title) for title in titles]

questions = [remove_punctuation(question) for question in questions]
questions_tokens = [nltk.word_tokenize(question) for question in questions]

titles_tokens = [nltk.word_tokenize(title) for title in titles]
stemmer = nltk.stem.PorterStemmer()

questions_stemmed = [[stemmer.stem(token) for token in tokens] for tokens in questions_tokens]

titles_stemmed = [[stemmer.stem(token) for token in tokens] for tokens in titles_tokens]
english_stopwords = set([word for word in nltk.corpus.stopwords.words("english")])

questions_stemmed = [[token for token in tokens if token not in english_stopwords] for tokens in questions_stemmed]

titles_stemmed = [[token for token in tokens if token not in english_stopwords] for tokens in titles_stemmed]

word_counts = pd.Series(np.concatenate(questions_stemmed)).value_counts()

singular_words = set(word_counts[pd.Series(np.concatenate(questions_stemmed)).value_counts() == 1].index)

questions_stemmed = [[word for word in title if word not in singular_words] for title in questions_stemmed]



#word_counts = pd.Series(np.concatenate(titles_stemmed)).value_counts()

#singular_words = set(word_counts[pd.Series(np.concatenate(titles_stemmed)).value_counts() == 1].index)

#titles_stemmed = [[word for word in title if word not in singular_words] for title in titles_stemmed]
non_empty_indices = [i for i in range(len(questions_stemmed)) if len(questions_stemmed[i]) > 0]

questions_stemmed = np.asarray(questions_stemmed)[non_empty_indices]

non_empty_indices = [i for i in range(len(titles_stemmed)) if len(titles_stemmed[i]) > 0]

titles_stemmed = np.asarray(titles_stemmed)[non_empty_indices]

dictionary = gensim.corpora.Dictionary(questions_stemmed)

dictionary2 = gensim.corpora.Dictionary(titles_stemmed)

corpus = [dictionary.doc2bow(text) for text in questions_stemmed]

corpus2 = [dictionary2.doc2bow(text) for text in titles_stemmed]

model =  gensim.models.ldamodel.LdaModel

ldamodel = model(corpus2, num_topics=5, id2word = dictionary2, passes=50)
print(ldamodel.print_topics(num_topics=5, num_words=10))