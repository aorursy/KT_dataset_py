import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

import gensim

from gensim.models import Word2Vec

from gensim.utils import simple_preprocess

from gensim.models import FastText

from gensim.models import TfidfModel

from gensim.corpora import Dictionary

from sklearn.neighbors import NearestNeighbors
df = pd.read_csv('../input/Questions.csv', encoding = "ISO-8859-1", nrows=30000, usecols=['Id', 'Title', 'Body'])

df.shape
#Let's take a look at some of the questions

print('Question1: ', df.iloc[0, 2])

print('Question2: ', df.iloc[1, 2])

print('Question3: ', df.iloc[2, 2])
#Using beautiful soup to grab text inside 'p' tags and concatenate it

def get_question(html_text):

  soup = BeautifulSoup(html_text, 'lxml')

  question = ' '.join([t.text for t in soup.find_all('p')]) #concatenating all p tags

  return question



#Transforming questions to list for ease of processing

question_list = df['Body'].apply(get_question).values.tolist()
question_list[0]
#Tokenizing with simple preprocess gensim's simple preprocess

def sent_to_words(sentences):

    for sentence in sentences:

        yield(simple_preprocess(str(sentence), deacc=True)) # returns lowercase tokens, ignoring tokens that are too short or too long



question_words = list(sent_to_words(question_list))
question_words[0][0:5] #first 5 question tokens
lengths = [len(question) for question in question_words]

plt.hist(lengths, bins = 25)

plt.show()



print('Mean word count of questions is %s' % np.mean(lengths))
#Getting rid of stopwords

stop_words = stopwords.words('english')



def remove_stopwords(sentence):

  filtered_words = [word for word in sentence if word not in stop_words]

  return filtered_words



filtered_questions = [remove_stopwords(question) for question in question_words]
#Examining word counts after removal of stop words



lengths = [len(question) for question in filtered_questions]

plt.hist(lengths, bins = 25)

plt.show()



print('Mean word count of questions is %s' % np.mean(lengths))
len(filtered_questions)
#Instantiating the model

n = 50

model = Word2Vec(filtered_questions, size = n, window = 8)



#Training model using questions corpora

model.train(filtered_questions, total_examples=len(filtered_questions), epochs=10)
#Let's see how it worked

word_vectors = model.wv



print('Words similar to "array" are: ', word_vectors.most_similar(positive='array'))

print('Words similar to "database" are: ', word_vectors.most_similar(positive='database'))
ft_model = FastText(filtered_questions, size=n, window=8, min_count=5, workers=2,sg=1)
print('Words similar to "array" are: ', ft_model.wv.most_similar('array'))

print('Words similar to "database" are: ', ft_model.wv.most_similar('database'))
#dct = Dictionary(filtered_questions)  # fit dictionary

#corpus = [dct.doc2bow(line) for line in filtered_questions]  # convert corpus to BoW format

#tfidf_model = TfidfModel(corpus)  # fit model
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(question_list)

print(X.shape)
#To proprely work with scikit's vectorizer

merged_questions = [' '.join(question) for question in filtered_questions]

document_names = ['Doc {:d}'.format(i) for i in range(len(merged_questions))]



def get_tfidf(docs, ngram_range=(1,1), index=None):

    vect = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)

    tfidf = vect.fit_transform(docs).todense()

    return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T



tfidf = get_tfidf(merged_questions, ngram_range=(1,1), index=document_names)
def get_sent_embs(emb_model):

    sent_embs = []

    for desc in range(len(filtered_questions)):

        sent_emb = np.zeros((1, n))

        if len(filtered_questions[desc]) > 0:

            sent_emb = np.zeros((1, n))

            div = 0

            model = emb_model

            for word in filtered_questions[desc]:

                if word in model.wv.vocab and word in tfidf.index:

                    word_emb = model.wv[word]

                    weight = tfidf.loc[word, 'Doc {:d}'.format(desc)]

                    sent_emb = np.add(sent_emb, word_emb * weight)

                    div += weight

                else:

                    div += 1e-13 #to avoid dividing by 0

        if div == 0:

            print(desc)



        sent_emb = np.divide(sent_emb, div)

        sent_embs.append(sent_emb.flatten())

    return sent_embs
ft_sent = get_sent_embs(emb_model = ft_model) 
def get_n_most_similar(interest_index, embeddings, n):

    """

    Takes the embedding vector of interest, the list with all embeddings, and the number of similar questions to 

    retrieve.

    Outputs the disctionary IDs and distances

    """

    nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)

    distances, indices = nbrs.kneighbors(embeddings)

    similar_indices = indices[interest_index][1:]

    similar_distances = distances[interest_index][1:]

    return similar_indices, similar_distances



def print_similar(interest_index, embeddings, n):

    """

    Convenience function for visual analysis

    """

    closest_ind, closest_dist = get_n_most_similar(interest_index, embeddings, n)

    print('Question %s \n \n is most similar to these %s questions: \n' % (question_list[interest_index], n))

    for question in closest_ind:

        print('ID ', question, ': ',question_list[question])
print_similar(42, ft_sent, 5)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(filtered_questions)]

model = Doc2Vec(documents, vector_size=n, window=8, min_count=5, workers=2, dm = 1, epochs=20)
print(question_list[42], ' \nis similar to \n')

print([question_list[similar[0]] for similar in model.docvecs.most_similar(42)])
print_similar(101, ft_sent, 5)