import numpy as np

import pandas as pd



import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 



from collections import Counter
data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
stopwords = set(stopwords.words('english'))

stopwords_dict = Counter(stopwords)



def preprocess_text(text):

    text = text.lower() # Convert to lowercase

    review = re.sub('[^a-zA-Z]',' ', text) # Remove words with non-letter characters

    words = text.split()

    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords_dict] # Remove stop words

    text = " ".join(words)

    return text
data['newDescription'] = data.description.apply(preprocess_text)
import gensim.downloader as api

model = api.load('word2vec-google-news-300')
# user has watched a title

pick = "Avengers: Infinity War"



pick_row = data[data.title.str.lower() == pick.lower()]

pick_index = pick_row.index.values[0]

pick_description = pick_row.newDescription.values[0]
# get wm distance

def getWMD(newText):

    return model.wmdistance(pick_description, newText)



# select nearest 10

def getTopNByWmd(df, col, n):

    return df.sort_values(by = col).head(n)
# compute wm distances

filteredData = data[data.index != pick_index]

filteredData['wmd'] = filteredData.newDescription.apply(getWMD)
getTopNByWmd(filteredData, 'wmd', 10).title
# Tokenize the documents.

from nltk.tokenize import RegexpTokenizer



docs = data.newDescription.copy()



# Split the documents into tokens.

tokenizer = RegexpTokenizer(r'\w+')

for idx in range(len(docs)):

    docs[idx] = docs[idx].lower()  # Convert to lowercase.

    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    

# Remove words that are only one character.

docs = [[token for token in doc if len(token) > 1] for doc in docs]
# Compute bigrams.

from gensim.models import Phrases



# Add bigrams and trigrams to docs (only ones that appear 20 times or more).

bigram = Phrases(docs, min_count=5, threshold=10)

for idx in range(len(docs)):

    for token in bigram[docs[idx]]:

        if '_' in token:

            # Token is a bigram, add to document.

            docs[idx].append(token)
from gensim.corpora import Dictionary



# Create a dictionary representation of the documents.

dictionary = Dictionary(docs)



# Filter out words that occur less than 20 documents, or more than 50% of the documents.

dictionary.filter_extremes(no_below=20, no_above=0.5)
# Bag-of-words representation of the documents.

corpus = [dictionary.doc2bow(doc) for doc in docs]
print('Number of unique tokens: %d' % len(dictionary))

print('Number of documents: %d' % len(corpus))
# Train LDA model.

from gensim.models import LdaModel, LdaMulticore



# Set training parameters.

num_topics = 15

chunksize = 2000

passes = 20

iterations = 100

eval_every = None  # Don't evaluate model perplexity, takes too much time.



# Make a index to word dictionary.

temp = dictionary[0]  # This is only to "load" the dictionary.

id2word = dictionary.id2token
from gensim.models import CoherenceModel



topic_size = [1,5,10,15,20,25,30,35,40]

coherence_score = []



def getModelCoherence(n_topics):

    sample_model = LdaMulticore(corpus=corpus,

                     id2word=id2word,

                     num_topics=n_topics,

                     chunksize=chunksize,

                     passes=passes,

                     iterations=10,

                     per_word_topics=True)

    

    cm = CoherenceModel(model=sample_model, corpus=corpus, dictionary=dictionary, coherence="u_mass")

    return cm.get_coherence()



for i in topic_size:

    coherence_score.append(getModelCoherence(i))
import matplotlib.pyplot as plt



plt.figure(figsize=(16, 8))

plt.plot(topic_size, coherence_score)



plt.title("Optimal LDA Model")

plt.xlabel("Number of Topics")

plt.ylabel("Coherence Scores")

plt.show()
num_topics = 10



model = LdaMulticore(corpus=corpus,

                     id2word=id2word,

                     num_topics=num_topics, 

                     chunksize=chunksize,

                     passes=passes,

                     random_state=80,

                     iterations=iterations,

                     per_word_topics=True)
for index, row in data.iterrows():

    for i in range(0,num_topics):

        data.at[index,'topic_'+str(i)] = 0

    for t in model.get_document_topics(corpus[index]):

        data.at[index,'topic_'+str(t[0])] = t[1]
# user has watched a title

pick = "Avengers: Infinity War"



pick_row = data[data.title.str.lower() == pick.lower()]

pick_index = pick_row.index.values[0]
# get wm distance

def Euclidean(row, n_topics):

    pick_vec = []

    row_vec = []

    for i in range(0,n_topics):

        pick_vec.append(pick_row.iloc[0]['topic_'+str(i)])

        row_vec.append(row['topic_'+str(i)])

        

    # Get similarity based on top k topics of picked vector

    k=10

    

    top_5_idx = np.argsort(pick_vec)[-k:]

    pick_vec = np.array(pick_vec)[top_5_idx]

    row_vec = np.array(row_vec)[top_5_idx]

    

    return np.linalg.norm(row_vec - pick_vec)



# select nearest 10

def getTopNByLDA(df, col, n):

    return df.sort_values(by = col).head(n)
# compute lda distances

filteredData = data.copy()

for index, row in filteredData.iterrows():

    filteredData.at[index,'lda'] = Euclidean(filteredData.iloc[index], num_topics)

    

filteredData = filteredData[filteredData.index != pick_index]
getTopNByLDA(filteredData, 'lda', 10).title