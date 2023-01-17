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

# Any results you write to the current directory are saved as output.
#importing libraries
import warnings
import numpy as np
import pandas as pd

import langdetect
from langdetect import detect
import string 
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS 
from spacy.lang.en import English

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings("ignore")#not show warning for deprecated
# read data
df=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
df.shape
df.info()
df.isnull().sum()
abstract=df["abstract"].dropna()#drooping all rows with all NaN values in abstract columns

len(abstract)
# fuction to make a mask selecting text by languaje
def detect_lang(text,lang):
    try:
        return detect(text['abstract']) == lang
    except:
        return False
    
# bulding a mask to filter out text written in English
df_abstracts = pd.DataFrame(abstract)
en_abstracts_mask = df_abstracts.apply(lambda row: detect_lang(row, "en"), axis=1)

abstracts_en=df_abstracts[en_abstracts_mask]
print(len(abstracts_en), "of our initial abstracts are writing in english languaje, we will use just those papers to made the analysis")
# Join the different processed titles together.
long_string = ','.join(list(abstracts_en["abstract"].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()
# list of punctuation and simbols
symbols_punctuations = string.punctuation

#  disabling Named Entity Recognition for speed
nlp = spacy.load('en',disable=['parser', 'ner'])

# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def tokenizer(text):
    # Creating our token object
    mytokens = nlp(text)

    # Lemmatizing each token and converting each token into lowercase if not pronoum
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words and puntuation
    mytokens = [ word for word in mytokens if word not in symbols_punctuations  and word not in stop_words]
 
    mytokens = [ word for word in mytokens if len(word) > 2]

    
    return mytokens
np.random.seed()
#creation of the bag of word with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# bag of words using our tokenizer .defining  ngram_range=(1,1) means only unigrams 

CountVectorizer_bow = CountVectorizer(tokenizer = tokenizer, ngram_range=(1,1),max_df=0.80,min_df=2) 
bow_raw_count=CountVectorizer_bow.fit_transform(abstracts_en["abstract"])
bow_raw_count.shape

"""from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# initializing the LDA object
# online Method used to update _component much faster
#0.5, 1.0] to guarantee asymptotic convergence for online method
LDA = LatentDirichletAllocation(max_iter=10, learning_method='online', learning_offset=50,random_state=0,batch_size=200,n_jobs=-1)

# Define Search Param
parameters = {"n_components": [7, 6, 9, 13, 16],"learning_decay": [0.5,0.7,0.9]}


# initializing gridsearchcv
grid_cv = GridSearchCV(LDA, param_grid=parameters)"""
# in a previus exploratory study we use the GridSearchCV to tunning the parameters as in the cell above, An we get the best reults for a learning_decay of 0.7 and 7 topics.

LDA = LatentDirichletAllocation(n_components=7, max_iter=10, learning_method='online', learning_offset=50,random_state=0,batch_size=200,n_jobs=-1,learning_decay=0.7)
LDA_topics=LDA.fit(bow_raw_count)

# Perplexity
print("Model Perplexity: ", LDA_topics.perplexity(bow_raw_count))

# Best score
print("Best Score: ", LDA_topics.score(bow_raw_count))

def display_topics(model, feature_label, no_top_words):

    for id, topic in enumerate(model.components_):
        print( "Topic:", (id))# print fisrt topic label
        print(" ".join([feature_label[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))# get sorted higher frecuency terms

labels=CountVectorizer_bow.get_feature_names()
display_topics(LDA_topics, labels, 10)#
# getting the topic and term performance for LDA

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(LDA_topics, bow_raw_count, CountVectorizer_bow, mds='tsne',n_jobs=-1)
panel
topic_values = LDA_topics.transform(bow_raw_count)
abstracts_en['topic_LDA'] = topic_values.argmax(axis=1)

abstracts_en.head(2)
document_by_topics_LDA=abstracts_en["topic_LDA"].value_counts()
document_by_topics_LDA_df=pd. DataFrame(document_by_topics_LDA).reset_index()
document_by_topics_LDA_df.columns=["topic_LDA","number_documents"]
document_by_topics_LDA_df

fig=plt.figure(figsize=(7,7))


plt.barh(document_by_topics_LDA_df["topic_LDA"],document_by_topics_LDA_df["number_documents"])
plt.ylabel("topic_LDA")
plt.xlabel("number_documents")
plt.title("Number of document by topic selected using LDA")

plt.show()

