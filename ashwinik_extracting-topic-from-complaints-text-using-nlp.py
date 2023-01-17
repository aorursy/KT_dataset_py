import pandas as pd
%matplotlib inline
from matplotlib import pyplot as plt
import nltk
import spacy
import seaborn as sns
import pickle
from nltk.stem import PorterStemmer
import string
complaint_data = pd.read_csv("../input/consumer-complaints-financial-products/Consumer_Complaints.csv",low_memory = False)
### Convert the columns names so that they don't have space and are more readable
complaint_data.columns = [i.lower().replace(" ","_").replace("-","_") for i in complaint_data.columns]
complaint_data.columns
print ("The shape of data is ",complaint_data.shape)
print ("The data types for our data are as follows ")

### All the varables are text - which may correspond to categories and other variables
print (" The number of unique values in each column is as follows")
### Lets do a describe with including objects
complaint_data.describe(include = 'object').T.reset_index()

#### Keep only the consumer complaints is not null
complaint_data = complaint_data[~complaint_data['consumer_complaint_narrative'].isna()]
#### Create a distirbution of length of customers complaints. We have very left skew in length of complaints
### Which is expected as most compalints can be written in less than 500 words
complaint_data['consumer_complaint_narrative'].apply(len).plot(kind = 'hist')
### Keep the length columns as a new column
complaint_data['comp_length'] = complaint_data['consumer_complaint_narrative'].apply(len)
complaint_data.reset_index(inplace = True)
complaint_data['consumer_complaint_narrative'] = complaint_data['consumer_complaint_narrative'].str.replace(r"XX+\s","")
complaint_data['consumer_complaint_narrative'] = complaint_data['consumer_complaint_narrative'].str.replace("XXXX","")
complaint_data = complaint_data.sample(frac = 0.5,random_state = 5)
complaint_data.reset_index(inplace = True)
nlp = spacy.load('en_core_web_sm')
### Remove the stop words using spacy predefined list 
stop_words = nlp.Defaults.stop_words
#### Create a list of puntuation to be removed
symbols = " ".join(string.punctuation).split(" ") 
### As we are doing topic modelling itsa good idea to do lemmatisation - as it uses morphologial analysis
ps = PorterStemmer()
import re
#### Lets define the cleaning function and see how it works
def cleanup_text(docs,logging = False):
    texts = []
    counter = 1
    for doc in docs:
        
        if counter % 5000 == 0 :
            print ("Processed %d of out of %d documents"% (counter,len(docs)))
        counter += 1
        
        doc = nlp(doc) ### We are disabling parser as will nt be using it
        
        
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != "-PRON-"]
        
        tokens =[tok for tok in tokens if tok not in symbols]
        tokens = [tok for tok in tokens if tok not in stop_words]
        tokens = [re.sub('[0-9]', '', i) for i in tokens]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return (pd.Series(texts))
complaint_data['comp_preprocessed'] = cleanup_text(complaint_data['consumer_complaint_narrative'])
print ("Shape of data before removing NA's ,",complaint_data.shape)
complaint_data =complaint_data[~complaint_data['comp_preprocessed'].isna()]
print ("Shape of data before removing NA's ,",complaint_data.shape)
complaint_data[['comp_preprocessed','consumer_complaint_narrative']].head(5)
### Lets Create the piprle line for NMF models 
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF 
#####  Let extract from act the features from the dataset

print ("Extracting the tf-idf features form NMF")
tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 5, max_features = 500, ngram_range = (1,4))

t0 = time()
tfidf = tfidf_vectorizer.fit_transform(complaint_data['comp_preprocessed'])
print ("done in %0.3fs." % (time() - t0))
#### Now we will fit model for 10 diff values of clusters
n_comp = [10,20,30,40,50,60,70,80,90,100,110]
loss = []
for comps in n_comp:
  
    t0 = time()
    nmf = NMF(n_components = comps, random_state = 1, beta_loss = 'kullback-leibler',solver = 'mu',max_iter = 200,
             alpha = 0.1, l1_ratio = 0.5).fit(tfidf)
    loss.append(nmf.reconstruction_err_)
    print ("done in %0.3f " % (time() -t0))
### Let try to create a elbow and find out the best model clusters
plt.plot(loss)
plt.xlabel('Number of Topics')
plt.ylabel('Reconstruction Error - Frobenius Norm')
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
print_top_words(nmf, tfidf_vectorizer.get_feature_names(), 10)
from sklearn.decomposition import LatentDirichletAllocation
# Fit the NMF model
#### Now we will fit model for 10 diff values of clusters
n_comp = [10,20,30,40,50]

for comps in n_comp:
    loss1 = []
    t0 = time()
    lda = LatentDirichletAllocation(n_components=comps, max_iter=2,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
print_top_words(lda, tfidf_vectorizer.get_feature_names(), 10)