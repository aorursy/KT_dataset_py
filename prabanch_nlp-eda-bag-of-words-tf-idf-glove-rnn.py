import string

import re

from os import listdir

from numpy import array

from nltk.corpus import stopwords

from collections import Counter

# Scikit Learn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_colwidth', -1)

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve

import numpy as np

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from keras.preprocessing.sequence import pad_sequences



import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, Dense, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras.models import Sequential

from keras import initializers, regularizers, constraints, optimizers, layers

from tensorflow.keras.callbacks import EarlyStopping

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



print("Libraries loaded")
path = '../input/nlp-getting-started/'

train = pd.read_csv(path + "train.csv")

test = pd.read_csv(path + "test.csv")

train['actual_text'] = train['text']
test.isnull().sum()
train[train.target==0]['text']
train[train.target==1]['text']
# train = train.head(10)?

test_df = test
train["target"].value_counts() / len(train) *100
import seaborn as sns

ax = sns.countplot(x='target', data=train)

ax.set_xlabel('target')

ax.set_ylabel("count")  
stopwords = set(STOPWORDS)





wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train[train.target==0]['text']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
stopwords = set(STOPWORDS)



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(train[train.target==1]['text']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=300)
from collections import defaultdict

train1_df = train[train["target"]==1]

train0_df = train[train["target"]==0]



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of Non Diaster Tweets", 

                                          "Frequent words of real Diaster Tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()

freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams of Non Diaster Tweets", 

                                          "Frequent bigrams of real Diaster Tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')

## Number of words in the text ##

train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



## Number of punctuations in the text ##

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Truncate some extreme values for better visuals ##

train['num_words'].loc[train['num_words']>60] = 60 #truncation for better visuals

train['num_punctuations'].loc[train['num_punctuations']>10] = 10 #truncation for better visuals

train['num_chars'].loc[train['num_chars']>350] = 350 #truncation for better visuals



f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='target', y='num_words', data=train, ax=axes[0])

axes[0].set_xlabel('Target', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='target', y='num_chars', data=train, ax=axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='target', y='num_punctuations', data=train, ax=axes[2])

axes[2].set_xlabel('Target', fontsize=12)

#plt.ylabel('Number of punctuations in text', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)





train['text']=train['text'].apply(lambda x : remove_URL(x))

train['text']=train['text'].apply(lambda x : remove_html(x))

train['text']=train['text'].apply(lambda x : remove_punct(x))





test_df['text']=test_df['text'].apply(lambda x : remove_URL(x))

test_df['text']=test_df['text'].apply(lambda x : remove_html(x))

test_df['text']=test_df['text'].apply(lambda x : remove_punct(x))



train['text']=train['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

test_df['text']=test_df['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))





import warnings; warnings.simplefilter('ignore')

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

stop = stopwords.words('english')



train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# train['text'] = train['text'].str.replace('\d+', '')

test_df['text'] = test_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# test_df['text'] = test_df['text'].str.replace('\d+', '')







stemmer = SnowballStemmer('english')

train['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

test_df['text'] = test_df['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
## split to train and val

train_df, val_df = train_test_split(train, test_size=0.1, random_state=2018)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve, auc,  f1_score

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import f1_score

# Making the Confusion Matrix

def get_metrics(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)



    class_names=[0,1] # name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')





    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



    # Model Precision: what percentage of positive tuples are labeled as such?

    print("Precision:",metrics.precision_score(y_test, y_pred))



    # Model Recall: what percentage of positive tuples are labelled as such?

    print(" True positive rate or (Recall or Sensitivity) :",metrics.recall_score(y_test, y_pred))



    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn+fp)



    #Specitivity. or True negative rate

    print(" True Negative rate or Specitivity :",specificity)



    false_negative = fn / (fn+tp)



    #False negative rate

    print(" False Negative rate :",false_negative)



    #False positive rate

    print(" False positive rate (Type 1 error) :",1 - specificity)

    

    print('F Score', f1_score(y_test, y_pred))

    print(cm)

# ## some config values 

# Get the tfidf vectors #

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.svm import LinearSVC



from sklearn.model_selection import GridSearchCV





train_X = train_df["text"].fillna("_na_")

val_X = val_df["text"].fillna("_na_")

test_X = test_df["text"].fillna("_na_")



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values



tfidf_vec = TfidfVectorizer(stop_words='english')

tfidf_vec.fit_transform(train_X.values.tolist()+val_X.values.tolist())

train_tfidf = tfidf_vec.transform(train_X.values.tolist())

val_tfidf = tfidf_vec.transform(val_X.values.tolist())

test_tfidf = tfidf_vec.transform(test_X.values.tolist())

print("tfidf done")





import numpy as np

from sklearn.dummy import DummyClassifier



dummy_clf = DummyClassifier(strategy="most_frequent")

print(dummy_clf)

dummy_clf.fit(train_tfidf, train_y)



y_dummy_clf = dummy_clf.predict(val_tfidf)



get_metrics(val_y, y_dummy_clf)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier( n_estimators=100, max_depth=5,    

                            min_samples_leaf=10, min_samples_split=20,

                            random_state=10)

grid_param = {

    'max_depth': [4,5],

    'min_samples_leaf': [10,20]

}



gd_sr = GridSearchCV(estimator=rf,

                     param_grid=grid_param,

                     scoring='accuracy',

                     cv=3,

                     n_jobs=-1)



gd_sr.fit(train_tfidf, train_y)



y_pred_rf = gd_sr.predict(val_tfidf)





get_metrics(val_y, y_pred_rf)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()





classifier.fit(train_tfidf, train_y)



y_pred_val_lr = classifier.predict(val_tfidf)



get_metrics(val_y, y_pred_val_lr)



import eli5

eli5.show_weights(classifier, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')
# from sklearn import svm



# model = svm.SVC(kernel='linear', probability=True)

# grid_param = {

#     'C': [1],

#     'gamma': [0.001]

# }



# gd_sr = GridSearchCV(estimator=model,

#                      param_grid=grid_param,

#                      scoring='accuracy',

#                      cv=3,

#                      n_jobs=-1)



# gd_sr.fit(train_tfidf, train_y)



# y_pred_val_svm = gd_sr.predict(val_tfidf)



# get_metrics(val_y, y_pred_val_svm)

# print(gd_sr.best_params_)
temp = val_df

temp.reset_index(drop=True, inplace=True)

forReview =pd.concat([temp, pd.DataFrame(y_pred_val_lr)],axis=1,ignore_index=True)

forReview= forReview.iloc[:,[3,4,5,14]]

forReview.columns = ['text','target','actual_text','predicted']

forReview
forReview[(forReview['target']==0) & (forReview['predicted']==1)]
forReview[(forReview['target']==1) & (forReview['predicted']==0)]
# # define the model

# model = Sequential()

# model.add(Dense(1024, input_dim=train_tfidf.shape[1]))

# model.add(Activation('relu'))



# model.add(Dense(1024))

# model.add(Activation('relu'))



# model.add(Dense(1))

# model.add(Activation('sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# print(model.summary())

# es = EarlyStopping(monitor='val_loss', patience=5)

# model.fit(train_tfidf,train_y, batch_size=512, epochs=1, validation_data=(val_tfidf, val_y), callbacks=[es])



# val_pred_y = model.predict_classes([val_tfidf], batch_size=1024, verbose=1)

# # val_pred_y = (val_pred_y>0.5).astype(int)

# get_metrics(val_y,val_pred_y)



# ## some config values 

# embed_size = 300 # how big is each word vector

# max_features = 5000 # how many unique words to use (i.e num rows in embedding vector)

# maxlen = 100 # max number of words in a question to use



# ## fill up the missing values

# train_X = train_df["text"].fillna("_na_").values

# val_X = val_df["text"].fillna("_na_").values

# test_X = test_df["text"].fillna("_na_").values



# ## Tokenize the sentences

# tokenizer = Tokenizer(num_words=max_features)

# tokenizer.fit_on_texts(list(train_X))

# train_X = tokenizer.texts_to_sequences(train_X)

# val_X = tokenizer.texts_to_sequences(val_X)

# test_X = tokenizer.texts_to_sequences(test_X)



# ## Pad the sentences 

# train_X = pad_sequences(train_X, maxlen=maxlen)

# val_X = pad_sequences(val_X, maxlen=maxlen)

# test_X = pad_sequences(test_X, maxlen=maxlen)



# ## Get the target values

# train_y = train_df['target'].values

# val_y = val_df['target'].values

# embedding_vecor_length = 32

# model = Sequential()

# model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))

# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))



# model.add(Dense(1024))

# model.add(Activation('relu'))

# model.add(Dropout(.2))



# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print(model.summary())

# es = EarlyStopping(monitor='val_loss', patience=10)

# model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks=[es])



# val_pred_y = model.predict_classes([val_X], batch_size=1024, verbose=1)

# get_metrics(val_y,val_pred_y)

# import numpy as np

# EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

# embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



# all_embs = np.stack(embeddings_index.values())

# emb_mean,emb_std = all_embs.mean(), all_embs.std()

# embed_size = all_embs.shape[1]



# word_index = tokenizer.word_index

# nb_words = min(max_features, len(word_index)) + 1

# embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

# for word, i in word_index.items():

#     if i >= max_features: continue

#     embedding_vector = embeddings_index.get(word)

#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        


# # define the model

# model = Sequential()



# e = Embedding(len(embedding_matrix), embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False)

# model.add(e)

# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

# model.add(Dense(1, activation='sigmoid'))

# # compile the model

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # summarize the model

# print(model.summary())

# es = EarlyStopping(monitor='val_loss', patience=10)

# model.fit(train_X, train_y, batch_size=512, epochs=10, validation_data=(val_X, val_y), callbacks=[es])



# val_pred_y = model.predict_classes([val_X], batch_size=1024, verbose=1)

# get_metrics(val_y,val_pred_y)



# # pred_glove_embed_y = model.predict([test_X], batch_size=1024, verbose=1)
# !pip install tokenizers
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import torch

import transformers as ppb

import warnings

warnings.filterwarnings('ignore')
path = '../input/nlp-getting-started/'

train = pd.read_csv(path + "train.csv")

test = pd.read_csv(path + "test.csv")



train = train[:2000]

## split to train and val

# train_df, val_df = train_test_split(train, test_size=0.2, random_state=2018)
# For DistilBERT:

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')



## Want BERT instead of distilBERT? Uncomment the following line:

#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')



# Load pretrained model/tokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
tokenized = train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
input_ids = torch.tensor(padded)  

attention_mask = torch.tensor(attention_mask)



with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
last_hidden_states
features = last_hidden_states[0][:,0,:].numpy()

labels = train.target

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
lr_clf = LogisticRegression()

lr_clf.fit(train_features, train_labels)

lr_clf.score(test_features, test_labels)