!pip install gensim
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, json
import gensim
import gensim.models.ldamodel
from gensim import corpora, utils
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet


filename = "/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json"
file =  open(filename, 'r')
for key, val in json.loads(file.readline()).items():
    print (key, val)
num_lines = sum(1 for line in file)
print("number of papers is : ", num_lines)
# A method to tokenize, lemmatize and removing stop words
import spacy
spacy.load('en')
from spacy.lang.en import English
lang_parser = English()
stops = set(stopwords.words('english'))

def get_valid_tokens(text):
    tokens = lang_parser(text)
    tokens = [token.text for token in tokens]
    tokens = [token for token in tokens if len(token)>4 and token not in stops]
    tokens = [wordnet.morphy(token) for token in tokens]
    return tokens
# Save only the first 1000 papers
papers = []
num_papers = 1000
with open(filename, 'r') as file:
    if num_papers:
        num_papers = num_papers - 1
        for paper in file:
            paper_descript = json.loads(paper)
            papers.append({"title":paper_descript["title"], "abstract": paper_descript["abstract"]})
            
import pickle
pickle.dump(papers, open('res_papers.pkl', 'wb'))
abstract_collection = []
papers = pickle.load(open('res_papers.pkl', 'rb'))
for paper in papers:
    abstract_collection = abstract_collection + get_valid_tokens(paper["abstract"])
pickle.dump(abstract_collection, open('abstracts.pkl', 'wb'))
len(abstract_collection)
abstract_collection = list(filter(None, abstract_collection))
abstract_collection = [abstract_collection]
print(abstract_collection)
dictionary = corpora.Dictionary(abstract_collection)
corpus = [dictionary.doc2bow(word) for word in abstract_collection]
pickle.dump(corpus, open('lda_corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
# For the time being suppose we have 25 topics.
NUM_TOPICS = 10
LDA = LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes = 20)
# Save the model for use during recommendation phase
LDA.save('res_art_25.gensim')
topics = LDA.print_topics(num_words = 10)
for topic in topics:
    print(topic)
# Visualize all the 25 topics
import pyLDAvis.gensim as visual
display = visual.prepare(LDA, corpus, dictionary)
pyLDavis.display(display)
# Prepare topic distribution vector for each paper.
topic_distributions = []
for paper in papers:
    topic_distributions.append(LDA[paper["abstract"]])
# create and save topic distributions for each user
def create_topic_distribution_user(userId, user_topics):
    # Select 10 words from each topic and ask the user to rate.
    user_topic_dist = []
    lda = LdaModel.load('res_art_25.gensim')
    topics = lda.print_topics(num_words = 10)
    for i in range(len(topics)): 
        for keyword in topics[i]:
            print(keyword)
        user_topic_dist[i] = input("Please rate your preference for the given keywords from 1 to 10:")
    user_topics[userId] = user_topic_dist/sum(user_topic_dist)
    return True
    
def get_similarity(userId):
    similarity = []
    for i in range(len(topic_distributions)):
        dis = topic_distributions[i]
        user_dist = user_topics[userId]
        similarity.push_back(1-spatial.distance.cosine(dis, user_dist))
        return similarity
userid = input("Please enter your userid. Type NA if new user.")
user_topics = {}
if userid == "NA":
    import random
    userid = int(random.random()*100000)
    create_user_topic_distribution()
similarity = get_similarity(userid)

best_article_indices = sorted(range(len(similarity)), key = lambda sub: similarity[sub])[-N:]
print("The follwing articles may interest you.")
for i in best_article_indices:
    print("Title\n",papers[i]["title"], "\n")
    print("Abstract\n", papers[i]["abstract"])