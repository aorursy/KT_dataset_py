import pandas as pd
import numpy as np
import nltk
import re
import gensim
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import FreqDist
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
# import pyLDAvis
# import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
Data = pd.read_csv('data.csv')
Data.shape
Data.head()
# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

freq_words(Data['text'])
# remove unwanted characters, numbers and symbols
Data['text'] = Data['text'].str.replace("[^a-zA-Z#]", " ")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
Data['text'] = Data['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in Data['text']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]
freq_words(reviews, 35)
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output
tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[1])
reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[1]) # print lemmatized review
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

Data['reviews'] = reviews_3

freq_words(Data['reviews'], 35)

dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()
## Extracting Topic wise Words
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 25)])

topwords = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
topwords['Topic'] = topwords['Topic'].astype(str)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
topwords
##Assigning Topic Name
topwords['Topic_Name'] = np.where(topwords['Topic'].str.contains('0|1'),'glassdoor_reviews',
                                  np.where(topwords['Topic'] == '4','room_rentals',
                                           np.where(topwords['Topic'] == '2','Automobiles','tech_news')))
topwords
## Extracting Top words for each topic
glassdoor_reviews = topwords[topwords['Topic_Name'] == 'glassdoor_reviews']['Word'].to_list()
room_rentals = topwords[topwords['Topic_Name'] == 'room_rentals']['Word'].to_list()
tech_news = topwords[topwords['Topic_Name'] == 'tech_news']['Word'].to_list()
Automobiles = topwords[topwords['Topic_Name'] == 'Automobiles']['Word'].to_list()

print (glassdoor_reviews)
print (room_rentals)
print (tech_news)
print (Automobiles)
## Refining the existing word list for topic
glassdoor_reviews.remove('year')
glassdoor_reviews.remove('season')
glassdoor_reviews.remove('new')
room_rentals.remove('minute')
tech_news.remove('new')

Automobiles.remove('new')
Automobiles.remove('time')
Automobiles.remove('thing')
Automobiles.remove('year')
Automobiles.remove('subject')
Automobiles.append('oil')
glassdoor_reviews.append('pros')
room_rentals.append('homes')
room_rentals.append('condo')
print (Automobiles)
##Predicting the topic based on words identified
Data['topic'] = np.where(Data['reviews'].str.contains(f"({'|'.join(room_rentals)})"),'room_rentals',
                         np.where(Data['reviews'].str.contains(f"({'|'.join(glassdoor_reviews)})"),'glassdoor_reviews',
                                  np.where(Data['reviews'].str.contains(f"({'|'.join(tech_news)})"),'tech_news',
                                           np.where(Data['reviews'].str.contains(f"({'|'.join(Automobiles)})"),'Automobiles','sports_news'))))
                                  
Data
##Final Data
Data1 = Data[['Id','topic']].drop_duplicates()
Data1.to_csv('submission_sweta_singh_v1.csv')
