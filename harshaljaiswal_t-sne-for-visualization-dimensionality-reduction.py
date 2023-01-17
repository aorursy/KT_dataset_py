import os
print(os.listdir("../input"))
#importing essential libraries
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
# using the SQLite Table to read data.
con = sqlite3.connect('../input/database.sqlite')
#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3
filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, con) 

# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative
# Score values for both positive and negative reviews
filtered_data['Score'].value_counts()
#Sorting data according to ProductId in ascending order
sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape

#Checking to see how much % of data still remains
(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100

final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
# checking whether anything is NaN or not
final.isnull().sum().sum()
# Taking Only 2k sample points with 1707 positive and 293 negative
sample2K_data = final.sample(n=2000, random_state=101)
sample2K_data['Score'].value_counts()
# find sentences containing HTML tags
import re
i=0;
for sent in sample2K_data['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1;



import nltk
nltk.download('stopwords')

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
#Code for implementing step-by-step the checks mentioned in the pre-processing phase
# this code takes a while to run as it needs to run on 500k sentences.
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in sample2K_data['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (sample2K_data['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(sample2K_data['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1
sample2K_data.shape
sample2K_data['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 
sample2K_data['CleanedText']=sample2K_data['CleanedText'].str.decode("utf-8")
sample2K_data.shape
# Taking out the class variable or score into separate series.
target = sample2K_data['Score']
#BoW
count_vect = CountVectorizer() #in scikit-learn
final_counts = count_vect.fit_transform(sample2K_data['CleanedText'].values)
print("the type of count vectorizer ",type(final_counts))
print("the shape of out text BOW vectorizer ",final_counts.get_shape())
print("the number of unique words ", final_counts.get_shape()[1])
# Converting sparse matrix to Dataframe
df = pd.DataFrame(final_counts.toarray())
# Shape of Dataframe df
df.shape
# Standardizing the df values
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(df)
print(standardized_data.shape)
# Taking Standardized values into nd array
data_2k = standardized_data
# Making a t-SNE plot from sklearn version of t-SNE
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0,perplexity=30)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_2k)

# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, target)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
# Ploting the result of tsne
import seaborn as sn
sn.FacetGrid(tsne_df, hue="label", height=8,).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
# Making a tf_idf vector
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
final_tf_idf = tf_idf_vect.fit_transform(sample2K_data['CleanedText'].values)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words including both unigrams and bigrams ", final_tf_idf.get_shape()[1])
# features gives list of words 
features = tf_idf_vect.get_feature_names()
type(features)
# Converting sparse matrix to Dataframe
df1 = pd.DataFrame(final_tf_idf.toarray())
# Standardizing the df values
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(df1)
print(standardized_data.shape)
data_2k = standardized_data
# Making a t-SNE plot from sklearn version of t-SNE
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0,perplexity=30)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_2k)


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, target)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
# Ploting the result of tsne
import seaborn as sn
sn.FacetGrid(tsne_df, hue="label", height=8,).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
# Train your own Word2Vec model using your own text corpus
i=0
list_of_sent=[]
for sent in sample2K_data['CleanedText'].values:
    list_of_sent.append(sent.split())
print(sample2K_data['CleanedText'].values[0])
print("*****************************************************************")
print(list_of_sent[0])
# min_count = 5 considers only words that occured atleast 5 times
w2v_model=Word2Vec(list_of_sent,min_count=5,size=50, workers=3)
w2v_words = list(w2v_model.wv.vocab)
print("number of words that occured minimum 5 times ",len(w2v_words))
print("sample words ", w2v_words[0:50])
# average Word2Vec
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))
X = np.array(sent_vectors)
# Making a t-SNE plot from sklearn version of t-SNE
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0,perplexity=30)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(X)


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, target)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
import seaborn as sn
sn.FacetGrid(tsne_df, hue="label", height=8,).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
# TF-IDF weighted Word2Vec
tfidf_feat = tf_idf_vect.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_of_sent: # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_words:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tf_idf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
X = np.array(tfidf_sent_vectors)
# Making a t-SNE plot from sklearn version of t-SNE
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0,perplexity=30)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(X)


# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, target)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
import seaborn as sn
sn.FacetGrid(tsne_df, hue="label", height=8,).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()