

# Load Training Dataset



import re

import pandas as pd

import numpy as np



##################################################

# Replace with WordNet contextual synonym

##################################################



from nltk.corpus import wordnet



def get_syn(w):

    syns = wordnet.synsets(w)

    if len(syns)==0:

        return w

    

    syn = syns[0]

    lemmas = syn.lemmas()

    if len(lemmas)==0:

        return w

    

    l = lemmas[0]

    s = l.name()

    return s



##################################################

# Load Dataset

##################################################



dataset=pd.read_csv('/kaggle/input/basedataset/train.csv')

dataset=dataset.query('EssaySet==1')

print(list(dataset.columns.values)) #dataset header

print(dataset.head(10)) #first N rows







##################################################

# Clean dataset

# Convert to lower case

# Tokenize

##################################################



from nltk.tokenize import sent_tokenize, word_tokenize 



for i in np.arange(0, len(dataset)):

    doc = dataset['EssayText'][i]

    clean_doc = " ".join(re.sub(r"[^A-Za-z \—]+", " ", doc).split())

    clean_doc = clean_doc.lower()



    dataset['EssayText'][i]=word_tokenize(clean_doc)

print(dataset.head(10)) #first N rows





##################################################

# Remove Stopwords, Stemming and Lemmatize

# Replace word with contextual synonym

##################################################



import nltk

nltk.download('stopwords') # Stopwords

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

from nltk.stem import PorterStemmer      # Stemmer



stop_words = set(stopwords.words('english')) 

le=WordNetLemmatizer()

ps = PorterStemmer() 

for i in np.arange(0, len(dataset)):

    word_tokens=dataset['EssayText'][i]

    #tokens = [get_syn(le.lemmatize(w)) for w in word_tokens]

    tokens = [get_syn(le.lemmatize(ps.stem(w))) for w in word_tokens if not w in stop_words]

    dataset['EssayText'][i] = " ".join(tokens)

print(dataset.head(10)) #first N rows

   

##################################################

# Split Dataset into Train Data and Test Data

##################################################



from sklearn.model_selection import train_test_split



X = dataset['EssayText'].tolist()          # EssayText

Y = dataset['Score1'].tolist()             # Score1

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)



##################################################

# Extracting The Features And Creating The Document-Term-Matrix

##################################################



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



vectorizer = TfidfVectorizer(max_df=0.72, max_features=10000, min_df=0, norm='l1' , sublinear_tf =True, stop_words='english', use_idf=True) # to play with. min_df,max_df,max_features etc...

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)



##################################################

# Latent Semantic Analysis (LSA)

##################################################



from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Normalizer



svd = TruncatedSVD(100)

lsa = make_pipeline(svd, Normalizer(copy=False))



X_train_lsa = lsa.fit_transform(X_train_tfidf)

X_test_lsa = lsa.transform(X_test_tfidf)





##################################################

# Classifying tfidf vectors

##################################################



print("\nClassifying tfidf vectors...")



# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 

# and brute-force calculation of distances.



from sklearn.neighbors import KNeighborsClassifier

knn_tfidf = KNeighborsClassifier(n_neighbors=47, algorithm='brute', metric='manhattan', weights = 'distance')

knn_tfidf.fit(X_train_tfidf, y_train)



# Classify the test vectors.

p = knn_tfidf.predict(X_test_tfidf)



# Measure accuracy

numRight = 0;

for i in range(0,len(p)):

    if p[i] == y_test[i]:

        numRight += 1



print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))



##################################################

# Classifying LSA vectors

##################################################



print("\nClassifying LSA vectors...")





# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 

# and brute-force calculation of distances.

knn_lsa = KNeighborsClassifier(n_neighbors=47, algorithm='brute', metric='manhattan', weights = 'distance')

knn_lsa.fit(X_train_lsa, y_train)



# Classify the test vectors.

p = knn_lsa.predict(X_test_lsa)



# Measure accuracy

numRight = 0;

for i in range(0,len(p)):

    if p[i] == y_test[i]:

        numRight += 1



print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))






