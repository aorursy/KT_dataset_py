!pip install langdetect

!pip install scikit-plot

!pip install nltk
### Ignore warning

import warnings

warnings.filterwarnings("ignore")
### Packages de base

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().magic('matplotlib inline')

import time

from itertools import chain 

from multiprocessing import Pool

import gc

import sys

from scipy import sparse

from random import sample 



### Packages pour NLP

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

import nltk 

stop = set(stopwords.words('english'))

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer

stemmer = SnowballStemmer('english')

import nltk.tokenize as tokenize

import gensim

from wordcloud import WordCloud

from langdetect import detect



### Packages pour les plots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()



### Packages ML

from umap import UMAP

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.externals import joblib

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers import Dense, BatchNormalization, Dropout

from keras.optimizers import Adam

from sklearn.preprocessing import Normalizer

import scikitplot as skplt
import os

os.chdir('../input/dataset') # Set working directory
df_train = pd.read_csv('train.txt', sep="\t",header =None)

df_train = pd.DataFrame(df_train[0].apply(lambda x: x.split(None, 1)).tolist(), columns=['label', 'text'])
print('Shape of data :',df_train.shape)
df_train.describe(include='all')
df_train.info()
df_train.head(20)
df_train = df_train.drop_duplicates()
df_train['label'].unique()
print(df_train['label'][0])

df_train['text'][0]
print(df_train['label'][1])

df_train['text'][1]
print(df_train['label'][6])

df_train['text'][6]
print(df_train['label'][14])

df_train['text'][14]
### Rename label

df_train['label_new'] = 'good'

df_train['label_new'][df_train['label']=='__label__1']='bad'

df_train.drop('label',axis = 1,inplace=True)
sns.barplot(np.unique(df_train['label_new']),df_train.groupby('label_new').count().values[:,0])
df_train['len'] = df_train['text'].apply(lambda x: len(x.split()))
plt.figure(figsize = (12, 7))

sns.distplot(df_train['len'][df_train['label_new'] =='good'], hist = True, label = "good",)

sns.distplot(df_train['len'][df_train['label_new'] =='bad'], hist = True, label = "bad")

plt.legend(fontsize = 10)

plt.title("Length Distribution by Class", fontsize = 12)

plt.show()
sf_train_exclamation_mark = df_train[df_train['text'].apply(lambda x: '!' in x)]

sns.barplot(np.unique(sf_train_exclamation_mark['label_new']),sf_train_exclamation_mark.groupby('label_new').count().values[:,0])

plt.title('Exclamation_mark (n =' +str(sf_train_exclamation_mark.shape[0])+')')
sf_train_question_mark = df_train[df_train['text'].apply(lambda x: '?' in x)]

sns.barplot(np.unique(sf_train_question_mark['label_new']),sf_train_question_mark.groupby('label_new').count().values[:,0])

plt.title('Question_mark (n =' +str(sf_train_question_mark.shape[0])+')')
sf_train_emotion_unhappy = df_train[df_train['text'].apply(lambda x: ':(' in x)]

sns.barplot(np.unique(sf_train_emotion_unhappy['label_new']),sf_train_emotion_unhappy.groupby('label_new').count().values[:,0])

plt.title('Emotion_unhappy (n =' +str(sf_train_emotion_unhappy.shape[0])+')')
sf_train_emotion_happy = df_train[df_train['text'].apply(lambda x: ':)' in x)]

sns.barplot(np.unique(sf_train_emotion_happy['label_new']),sf_train_emotion_happy.groupby('label_new').count().values[:,0])

plt.title('Emotion_happy (n =' +str(sf_train_emotion_happy.shape[0])+')')
train_sentiment = pd.DataFrame.from_records(df_train['text'][0:10000].apply(lambda x: sia.polarity_scores(x)))

train_sentiment['len'] = df_train['text'][0:10000].apply(len)

train_sentiment['label'] = 'good'

train_sentiment['label'][train_sentiment['neg']>train_sentiment['pos']] ='bad'

train_sentiment['real_label'] = df_train['label_new'][0:10000]

train_sentiment.head()
print(classification_report(df_train['label_new'][0:10000], train_sentiment['label']))
accuracy_score(df_train['label_new'][0:10000], train_sentiment['label'])
lang_detection = df_train.iloc[0:10000]

lang_detection['lang'] = df_train['text'][0:10000].map(detect,) 
lang_detection[lang_detection['lang'] != 'en']
lang_detection['lang'].value_counts()
### Clean memory

n_obs = 10000

df_train = df_train.iloc[0:n_obs]

del sf_train_emotion_happy, sf_train_emotion_unhappy, sf_train_question_mark, sf_train_exclamation_mark, lang_detection

gc.collect()
from textblob import TextBlob

def lemmatize_with_postag(sentence):

    sent = TextBlob(sentence)

    tag_dict = {"J": 'a', 

                "N": 'n', 

                "V": 'v', 

                "R": 'r'}

    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    

    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

    return " ".join(lemmatized_list)



### Function which define if a word is a number or not



def is_number(word):

    try: 

        x = float(word)

        return (x == x) and (x - 1 != x)

    except Exception:

        return False

    

### Tokenization using regular expression pattern which keep numbers, ?_mark, !_mark, :), :(     

tokenizer= tokenize.RegexpTokenizer(r'[0-9]*\.?[0-9]+|[a-zA-Z]+|[!]+|[:)]+|[:(]+|[?]+|[^[a-zA-Z]\s]+|[^[a-zA-Z][0-9]]')



### dictionary number to word

number2word = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','.':'point'}



### English Stemming 

stemmer = SnowballStemmer('english')



def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    4. Keep ?, keep !

    5. Keep . for decimal numbers

    """

    mess = tokenizer.tokenize(mess.lower())

    

    word_list = []

        

    for word in mess:

        if (word ==':') or (word.startswith(')')) or (word.startswith('(')): 

            next

        elif word.startswith(':)') :

            word_list.append('emotion_happy')

        elif word.startswith(':(') :

            word_list.append('emotion_unhappy')

        elif word.startswith('?') : 

            word_list.append('question_mark')

        elif word.startswith('!') :

            word_list.append('exclamation_mark')

        elif (word not in stopwords.words('english')) and (word not in ':.%...') and (is_number(word)==False):

            word_list.append(lemmatize_with_postag(word))

        elif is_number(word)==True:

            b = [number2word.get(i) for i in list(word)]

            word_list += b

    return word_list



print('Pharse originale : \n')

print(df_train['text'][32],'\n')

print('Pharse transformée: \n')

print(' '.join(text_process(df_train['text'][32])),'\n')

print('Liste de mots (Tokenization): \n')

text_process(df_train['text'][32])
print('Pharse originale : \n')

print(df_train['text'][979],'\n')

print('Pharse transformée: \n')

print(' '.join(text_process(df_train['text'][979])),'\n')

print('Liste de mots (Tokenization): \n')

text_process(df_train['text'][32])
### These functions allow us to work in parallel

num_partitions = 10 #number of partitions to split dataframe

num_cores = 4 #number of cores on your machine



def vec_text_process(array):

    return array.apply(text_process)



def parallelize_dataframe(df):

    df_split = np.array_split(df, num_partitions)

    pool = Pool(num_cores)

    df = pd.Series(list(chain.from_iterable(pool.map(vec_text_process, df_split))))

    pool.close()

    pool.join()

    return df



df_train_tokens = parallelize_dataframe(df_train['text'][0:n_obs])
train_index = df_train.index.isin(sample(range(int(n_obs)), int(n_obs*0.7)))

X_train = df_train_tokens[train_index]

Y_train = df_train['label_new'][train_index].reset_index(drop=True)



X_test = df_train_tokens[~train_index].reset_index(drop=True)

Y_test = df_train['label_new'][~train_index].reset_index(drop=True)



### Clear memory

#del df_train, df_train_tokens

gc.collect()
### Create a dictionary 

df_train_dict = gensim.corpora.Dictionary(X_train)
### we draw word cloud on first 10000 reviews 

text = ' '.join(df_train['text'].str.lower().values[0:10000])

wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',

                      width=1200, height=1000).generate(text)

plt.figure(figsize=(12, 8))

plt.imshow(wordcloud)

plt.title('Top words in review')

plt.axis("off")

plt.show()
### Clear memory

del df_train, df_train_tokens

gc.collect()
# fonction qui sort la dataframe de la fréquence

def most_frequent_words(dictionary) : 

    df_tokens = pd.DataFrame.from_dict(dictionary.dfs, orient='index',columns=['counts'])

    get_words = np.vectorize(dictionary.get)

    df_tokens['word'] = get_words(df_tokens.index)

    df_tokens.sort_values('counts',ascending=False,inplace = True)

    return df_tokens

df_tokens = most_frequent_words(df_train_dict)
print('The length of dictionary =', df_tokens.shape[0])
df_tokens.reset_index(drop = True, inplace=True)

df_tokens.head()
fig, ax1 = plt.subplots(figsize=(14,8))

g = sns.barplot(df_tokens['word'][0:50],y=df_tokens['counts'][0:50], ax=ax1)

plt.xticks(rotation=90,fontsize=10)

plt.title('The most 50 frequent words')



ax2 = ax1.twinx()

ax2.set_ylim(0,ax1.get_ylim()[1]/X_train.shape[0])

ax2.set_ylabel('counts / documents')

fig.tight_layout() 

plt.show()
df_tokens['counts'].describe()
print('Quantile 95% :', np.quantile(df_tokens['counts'], 0.95))

print('Quantile 97.5% :', np.quantile(df_tokens['counts'], 0.975))

print('Number of words appearing only 1 time =', sum(df_tokens['counts'] ==1 ))

print('Number of frequencies of the 10000th word =', df_tokens['counts'][9999])
df_train_dict_1 = df_train_dict

df_train_dict_1.filter_extremes(keep_n=1000)

X_train_corpus  = [df_train_dict_1.doc2bow(doc) for doc in X_train]

X_train_visu = gensim.matutils.corpus2dense(X_train_corpus, num_terms=len(df_train_dict_1)).T

a = X_train_visu/ X_train_visu.sum(axis =0)[None,:]
plt.figure(figsize=(15,8))



plt.imshow(a[np.argsort(Y_train),:].T,)
df_train_dict.filter_extremes(keep_n=10000)
X_train_corpus  = [df_train_dict.doc2bow(doc) for doc in X_train]

X_train_tfidf = gensim.models.TfidfModel(X_train_corpus, df_train_dict)



### transform data to spare matrix in order to reduce memory

X_train_tfidf_full = gensim.matutils.corpus2csc(X_train_tfidf[X_train_corpus], num_terms=len(df_train_dict)).T
X_test_corpus  = [df_train_dict.doc2bow(doc) for doc in X_test]

X_test_tfidf = gensim.models.TfidfModel(X_test_corpus, df_train_dict)



### transform data to spare matrix in order to reduce memory

X_test_tfidf_full = gensim.matutils.corpus2csc(X_test_tfidf[X_test_corpus], num_terms=len(df_train_dict)).T
df_train_dict.filter_extremes(keep_n=1000)

X_train_corpus_visu  = [df_train_dict.doc2bow(doc) for doc in X_train]

X_train_tfidf_visu = gensim.models.TfidfModel(X_train_corpus_visu, df_train_dict)

X_train_tfidf_visu = gensim.matutils.corpus2dense(X_train_tfidf_visu[X_train_corpus_visu], num_terms=len(df_train_dict)).T

b = X_train_tfidf_visu/ X_train_tfidf_visu.sum(axis =0)[None,:]

plt.figure(figsize=(15,8))

plt.imshow(b[np.argsort(Y_train),:].T)
### For PCA, we try first with 200 components

p = time.time()

pca = PCA(n_components = 200)

X_train_tfidf_pca = pca.fit_transform(X_train_tfidf_full.A)

pca_analysis = pd.DataFrame(columns = ['Number of components','Cumulative of variance ratio'])

pca_analysis['Number of components'] = np.arange(1,201)

pca_analysis['Cumulative of variance ratio'] = np.cumsum(pca.explained_variance_ratio_)

pca_analysis
X_test_tfidf_pca = pca.transform(X_test_tfidf_full.A)

time_pca=time.time() - p
p = time.time()

lda_model_tfidf = gensim.models.LdaMulticore(X_train_tfidf[X_train_corpus], num_topics= 20, id2word=df_train_dict, passes=2, workers=4)

X_train_lda_topic = lda_model_tfidf[X_train_tfidf[X_train_corpus]]

X_train_tfidf_lda = sparse.lil_matrix((len(X_train_tfidf[X_train_corpus]),20), dtype=np.float64)



for i in range(len(X_train_tfidf[X_train_corpus])):

    for j in X_train_lda_topic[i]:

        X_train_tfidf_lda[i,j[0]] = j[1]



X_test_lda_topic = lda_model_tfidf[X_test_tfidf[X_test_corpus]]

X_test_tfidf_lda = sparse.lil_matrix((len(X_test_tfidf[X_test_corpus]),20), dtype=np.float64)



for i in range(len(X_test_tfidf[X_test_corpus])):

    for j in X_test_lda_topic[i]:

        X_test_tfidf_lda[i,j[0]] = j[1]

        

del lda_model_tfidf, X_train_lda_topic, X_test_lda_topic

time_lda=time.time() - p
p = time.time()

rp_model_tfidf = gensim.models.RpModel(X_train_tfidf[X_train_corpus], id2word=df_train_dict, num_topics= 100)

X_train_rp_topic = rp_model_tfidf[X_train_tfidf[X_train_corpus]]

X_train_tfidf_rp = sparse.lil_matrix((len(X_train_tfidf[X_train_corpus]),100), dtype=np.float64)



for i in range(len(X_train_tfidf[X_train_corpus])):

    for j in X_train_rp_topic[i]:

        X_train_tfidf_rp[i,j[0]] = j[1]

        



X_test_rp_topic = rp_model_tfidf[X_test_tfidf[X_test_corpus]]

X_test_tfidf_rp = sparse.lil_matrix((len(X_test_tfidf[X_test_corpus]),100), dtype=np.float64)



for i in range(len(X_test_tfidf[X_test_corpus])):

    for j in X_test_rp_topic[i]:

        X_test_tfidf_rp[i,j[0]] = j[1]

        

del rp_model_tfidf, X_train_rp_topic, X_test_rp_topic



time_rp=time.time() - p
### n_components can be set between 1 to 100, but this technique consume a lot of time, so we choose only 20

p = time.time()

umap = UMAP(n_components=20, metric = 'cosine')

X_train_tfidf_umap = umap.fit_transform(X_train_tfidf_full.A)

X_test_tfidf_umap = umap.transform(X_test_tfidf_full.A)

del umap

time_umap = time.time()-p
train_sentiment['len'] = Normalizer().fit_transform(np.array(train_sentiment['len']).reshape(1,-1)).reshape(-1,1)

sentiment = train_sentiment[['neg','neu','pos','compound','len']]

del train_sentiment



X_train_tfidf_full = sparse.hstack((sentiment[0:int(n_obs*0.7)], X_train_tfidf_full))

X_test_tfidf_full = sparse.hstack((sentiment[int(n_obs*0.7):n_obs], X_test_tfidf_full))



X_train_tfidf_pca = np.concatenate([sentiment[0:int(n_obs*0.7)],X_train_tfidf_pca],axis =1)

X_test_tfidf_pca = np.concatenate([sentiment[int(n_obs*0.7):n_obs],X_test_tfidf_pca],axis =1)



X_train_tfidf_lda = sparse.hstack((sentiment[0:int(n_obs*0.7)], X_train_tfidf_lda))

X_test_tfidf_lda = sparse.hstack((sentiment[int(n_obs*0.7):n_obs], X_test_tfidf_lda))



X_train_tfidf_rp = sparse.hstack((sentiment[0:int(n_obs*0.7)], X_train_tfidf_rp))

X_test_tfidf_rp = sparse.hstack((sentiment[int(n_obs*0.7):n_obs], X_test_tfidf_rp))



X_train_tfidf_umap = np.concatenate([sentiment[0:int(n_obs*0.7)],X_train_tfidf_umap],axis =1)

X_test_tfidf_umap = np.concatenate([sentiment[int(n_obs*0.7):n_obs],X_test_tfidf_umap],axis =1)



del sentiment
### Clean memory

del X_train, X_test
gc.collect()
dimension_reduction_comparison = pd.DataFrame(columns = ['Method','Computing time','Number of new features'])

dimension_reduction_comparison['Method']=['Principal component analysis (PCA)','Random projection (RP)','Latent Dirichet Allocation (LDA)','Uniform Manifold Approximation and Projection (UMAP)']

dimension_reduction_comparison['Computing time'] = [time_pca, time_lda, time_rp, time_umap]

dimension_reduction_comparison['Number of new features'] =[200,20,100,20]

dimension_reduction_comparison
### On full data

p = time.time()

model_LR_full = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_LR_full.fit(X_train_tfidf_full,Y_train)

Y_predict_LR_full = model_LR_full.predict(X_test_tfidf_full)

time_LR_full = time.time()-p
### On PCA data

p = time.time()

model_LR_pca = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_LR_pca.fit(X_train_tfidf_pca,Y_train)

Y_predict_LR_pca = model_LR_pca.predict(X_test_tfidf_pca)

time_LR_pca = time.time()-p
### On LDA data

p = time.time()

model_LR_lda = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_LR_lda.fit(X_train_tfidf_lda,Y_train)

Y_predict_LR_lda = model_LR_lda.predict(X_test_tfidf_lda)

time_LR_lda = time.time()-p
### On RP data

p = time.time()

model_LR_rp = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_LR_rp.fit(X_train_tfidf_rp,Y_train)

Y_predict_LR_rp = model_LR_rp.predict(X_test_tfidf_rp)

time_LR_rp = time.time()-p
### On UMAP data

p = time.time()

model_LR_umap = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

model_LR_umap.fit(X_train_tfidf_umap,Y_train)

Y_predict_LR_umap = model_LR_umap.predict(X_test_tfidf_umap)

time_LR_umap = time.time()-p
### On full data

p = time.time()

model_SVM_full = SVC(kernel='linear', probability = True)

model_SVM_full.fit(X_train_tfidf_full,Y_train)

Y_predict_SVM_full = model_SVM_full.predict(X_test_tfidf_full)

Y_predict_proba_SVM_full = model_SVM_full.predict_proba(X_test_tfidf_full)

time_SVM_full = time.time()-p
### On PCA data

p = time.time()

model_SVM_pca = SVC(kernel='linear', probability = True)

model_SVM_pca.fit(X_train_tfidf_pca,Y_train)

Y_predict_SVM_pca = model_SVM_pca.predict(X_test_tfidf_pca)

Y_predict_proba_SVM_pca = model_SVM_pca.predict_proba(X_test_tfidf_pca)

time_SVM_pca = time.time()-p
### On LDA data

p = time.time()

model_SVM_lda = SVC(kernel='linear')

model_SVM_lda.fit(X_train_tfidf_lda,Y_train)

Y_predict_SVM_lda = model_SVM_lda.predict(X_test_tfidf_lda)

time_SVM_lda = time.time()-p
### On RP data

p = time.time()

model_SVM_rp = SVC(kernel='linear')

model_SVM_rp.fit(X_train_tfidf_rp,Y_train)

Y_predict_SVM_rp = model_SVM_rp.predict(X_test_tfidf_rp)

time_SVM_rp = time.time()-p
### On UMAP data

p = time.time()

model_SVM_umap = SVC(kernel='linear')

model_SVM_umap.fit(X_train_tfidf_umap,Y_train)

Y_predict_SVM_umap = model_SVM_umap.predict(X_test_tfidf_umap)

time_SVM_umap = time.time()-p
### On full data

p = time.time()

model_KNN_full = KNeighborsClassifier()

model_KNN_full.fit(X_train_tfidf_full,Y_train)

Y_predict_KNN_full = model_KNN_full.predict(X_test_tfidf_full)

time_KNN_full = time.time()-p
### On PCA data

p = time.time()

model_KNN_pca = KNeighborsClassifier()

model_KNN_pca.fit(X_train_tfidf_pca,Y_train)

Y_predict_KNN_pca = model_KNN_pca.predict(X_test_tfidf_pca)

time_KNN_pca = time.time()-p
### On LDA data

p = time.time()

model_KNN_lda = KNeighborsClassifier()

model_KNN_lda.fit(X_train_tfidf_lda,Y_train)

Y_predict_KNN_lda = model_KNN_lda.predict(X_test_tfidf_lda)

time_KNN_lda = time.time()-p
### On RP data

p = time.time()

model_KNN_rp = KNeighborsClassifier()

model_KNN_rp.fit(X_train_tfidf_rp,Y_train)

Y_predict_KNN_rp = model_KNN_rp.predict(X_test_tfidf_rp)

time_KNN_rp = time.time()-p
### On UMAP data

p = time.time()

model_KNN_umap = KNeighborsClassifier()

model_KNN_umap.fit(X_train_tfidf_umap,Y_train)

Y_predict_KNN_umap = model_KNN_umap.predict(X_test_tfidf_umap)

time_KNN_umap = time.time()-p
Y_train_NN = (Y_train=='good')

Y_test_NN = (Y_test=='good')
### On full data

p = time.time()

model_NN = Sequential()

model_NN.add(Dense(128, activation = 'relu',input_shape=(X_train_tfidf_full.shape[1],)))

model_NN.add(BatchNormalization())

model_NN.add(Dropout(0.2)) # to avoid overfitting

model_NN.add(Dense(1, activation = 'sigmoid'))

model_NN.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.01), metrics = ['accuracy'])

model_NN.fit(X_train_tfidf_full.A,Y_train_NN, validation_data=(X_test_tfidf_full.A, Y_test_NN), epochs = 5, batch_size = 128)

Y_predict_NN_full = np.where(model_NN.predict(X_test_tfidf_full.A)<0.5,'bad', 'good')

time_NN_full = time.time()
### On PCA data

p = time.time()

model_NN = Sequential()

model_NN.add(Dense(128, activation = 'relu',input_shape=(X_train_tfidf_pca.shape[1],)))

model_NN.add(BatchNormalization())

model_NN.add(Dropout(0.2)) # to avoid overfitting

model_NN.add(Dense(1, activation = 'sigmoid'))

model_NN.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.01), metrics = ['accuracy'])

model_NN.fit(X_train_tfidf_pca,Y_train_NN, validation_data=(X_test_tfidf_pca, Y_test_NN), epochs = 5, batch_size = 128)

Y_predict_NN_pca = np.where(model_NN.predict(X_test_tfidf_pca)<0.5,'bad', 'good')

time_NN_pca = time.time()
### On LDA data

p = time.time()

model_NN = Sequential()

model_NN.add(Dense(128, activation = 'relu',input_shape=(X_train_tfidf_lda.shape[1],)))

model_NN.add(BatchNormalization())

model_NN.add(Dropout(0.2)) # to avoid overfitting

model_NN.add(Dense(1, activation = 'sigmoid'))

model_NN.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.01), metrics = ['accuracy'])

model_NN.fit(X_train_tfidf_lda.A,Y_train_NN, validation_data=(X_test_tfidf_lda.A, Y_test_NN), epochs = 5, batch_size = 128)

Y_predict_NN_lda = np.where(model_NN.predict(X_test_tfidf_lda.A)<0.5,'bad', 'good')

time_NN_lda = time.time()
### On RP data

p = time.time()

model_NN = Sequential()

model_NN.add(Dense(128, activation = 'relu',input_shape=(X_train_tfidf_rp.shape[1],)))

model_NN.add(BatchNormalization())

model_NN.add(Dropout(0.2)) # to avoid overfitting

model_NN.add(Dense(1, activation = 'sigmoid'))

model_NN.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.01), metrics = ['accuracy'])

model_NN.fit(X_train_tfidf_rp.A,Y_train_NN, validation_data=(X_test_tfidf_rp.A, Y_test_NN), epochs = 5, batch_size = 128)

Y_predict_NN_rp = np.where(model_NN.predict(X_test_tfidf_rp.A)<0.5,'bad', 'good')

time_NN_rp = time.time()
### On UMAP data

p = time.time()

model_NN = Sequential()

model_NN.add(Dense(128, activation = 'relu',input_shape=(X_train_tfidf_umap.shape[1],)))

model_NN.add(BatchNormalization())

model_NN.add(Dropout(0.2)) # to avoid overfitting

model_NN.add(Dense(1, activation = 'sigmoid'))

model_NN.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.01), metrics = ['accuracy'])

model_NN.fit(X_train_tfidf_umap,Y_train_NN, validation_data=(X_test_tfidf_umap, Y_test_NN), epochs = 5, batch_size = 128)

Y_predict_NN_umap = np.where(model_NN.predict(X_test_tfidf_umap)<0.5,'bad', 'good')

time_NN_umap = time.time()
def accuracy_score_vector(x) :

    return [accuracy_score(Y_test, predictor) for predictor in x]



perfomance_comparison =pd.DataFrame(columns = ['Machine Learning model','Reduction method','Accuracy score','Computing time'])

perfomance_comparison['Machine Learning model'] = ['Logistic Regression']*5 + ['Support vector machine']*5+['K-nearest neighbors']*5+['Neural network']*5

perfomance_comparison['Reduction method'] = ['Full matrix', 'PCA', 'LDA', 'RP', 'UMAP']*4

perfomance_comparison['Accuracy score'] = accuracy_score_vector([

    Y_predict_LR_full,Y_predict_LR_pca, Y_predict_LR_lda, Y_predict_LR_rp, Y_predict_LR_umap,

    Y_predict_SVM_full,Y_predict_SVM_pca, Y_predict_SVM_lda, Y_predict_SVM_rp, Y_predict_SVM_umap,

    Y_predict_KNN_full,Y_predict_KNN_pca, Y_predict_KNN_lda, Y_predict_KNN_rp, Y_predict_KNN_umap,

    Y_predict_NN_full,Y_predict_NN_pca, Y_predict_NN_lda, Y_predict_NN_rp, Y_predict_NN_umap])

perfomance_comparison['Computing time'] = [time_LR_full, time_LR_pca, time_LR_lda, time_LR_rp, time_LR_umap,

                                           time_SVM_full, time_SVM_pca, time_SVM_lda, time_SVM_rp, time_SVM_umap,

                                           time_KNN_full, time_KNN_pca, time_KNN_lda, time_KNN_rp, time_KNN_umap,

                                           time_NN_full, time_NN_pca, time_NN_lda, time_NN_rp, time_NN_umap]

perfomance_comparison['Computing time added dimension reduction time'] = perfomance_comparison['Computing time'] + np.array([0,time_pca, time_lda, time_rp, time_umap]*4)

perfomance_comparison['Estimated computing time on all training set'] = perfomance_comparison['Computing time added dimension reduction time']/n_obs*3600000

perfomance_comparison

print(classification_report(Y_test, Y_predict_LR_full))
cm = confusion_matrix(Y_test,Y_predict_LR_full)

cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

#Visualization of the Confusion matrix

ax=plt.subplot(title='Confusion matrix of the Logistic Model on full data set');

sns.heatmap(cm,annot=True, fmt='.4f', cmap='Blues', xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])

ax.set_xlabel('LR Prediction');

ax.set_ylabel('True labels');
Y_predict_proba_LR_full = model_LR_full.predict_proba(X_test_tfidf_full)

skplt.metrics.plot_roc_curve(Y_test, Y_predict_proba_LR_full)

plt.show()
print(classification_report(Y_test, Y_predict_LR_pca))
cm = confusion_matrix(Y_test,Y_predict_LR_pca)

cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

#Visualization of the Confusion matrix

ax=plt.subplot(title='Confusion matrix of the Logistic Model on PCA data set');

sns.heatmap(cm,annot=True, fmt='.4f', cmap='Blues', xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])

ax.set_xlabel('LR Prediction');

ax.set_ylabel('True labels');
Y_predict_proba_LR_pca = model_LR_pca.predict_proba(X_test_tfidf_pca)

skplt.metrics.plot_roc_curve(Y_test, Y_predict_proba_LR_pca)

plt.show()
print(classification_report(Y_test, Y_predict_SVM_full))
cm = confusion_matrix(Y_test,Y_predict_SVM_full)

cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

#Visualization of the Confusion matrix

ax=plt.subplot(title='Confusion matrix of the Support Vector Machines on full data set');

sns.heatmap(cm,annot=True, fmt='.4f', cmap='Blues', xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])

ax.set_xlabel('SVM Prediction');

ax.set_ylabel('True labels');
Y_predict_proba_SVM_full = model_SVM_full.predict_proba(X_test_tfidf_full)

skplt.metrics.plot_roc_curve(Y_test, Y_predict_proba_SVM_full)

plt.show()
print(classification_report(Y_test, Y_predict_SVM_pca))
cm = confusion_matrix(Y_test,Y_predict_SVM_pca)

cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

#Visualization of the Confusion matrix

ax=plt.subplot(title='Confusion matrix of the Support Vector Machines on PCA data set');

sns.heatmap(cm,annot=True, fmt='.4f', cmap='Blues', xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])

ax.set_xlabel('SVM Prediction');

ax.set_ylabel('True labels');
Y_predict_proba_SVM_pca = model_SVM_pca.predict_proba(X_test_tfidf_pca)

skplt.metrics.plot_roc_curve(Y_test, Y_predict_proba_SVM_pca)

plt.show()