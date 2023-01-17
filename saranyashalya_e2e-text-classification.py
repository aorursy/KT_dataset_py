!set KERAS_BACKEND=tensorflow
import string

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go





from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import linear_model, model_selection, metrics

import eli5







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

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers



!ls ../input/quora-insincere-questions-classification/
!ls ../input/quora-insincere-questions-classification/embeddings/
train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print(train_df.shape)

print(test_df.shape)
train_df.head()
cnt_srs = train_df.target.value_counts()

trace = go.Bar(

        x = cnt_srs.index,

        y = cnt_srs.values,

        marker = dict(

                    color = cnt_srs.values,

                    colorscale = 'Picnic',

                    reversescale = True

                    ),

            )



layout = go.Layout(

            title ='Target count',

            font = dict(size = 18)

            )



data = [trace]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = "TargetCount")
##Target distribution

labels = (np.array(cnt_srs.index))

sizes = np.array((cnt_srs/cnt_srs.sum())*100)
trace = go.Pie(labels = labels, values = sizes)

layout = go.Layout(

            title ='Target distribution',

            font = dict(size=18),

            height = 600,

            width = 600

            )

data = [trace]

fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = 'usertype')
from wordcloud import WordCloud, STOPWORDS
def plot_wordcloud(text, mask = None, max_words = 200, max_font_size = 100, figure_size= (24.0, 16.0),

                  title = None, title_size = 40, image_color = False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)

    

    wordcloud = WordCloud(background_color = 'black',

                        stopwords = stopwords,

                         max_words = max_words,

                         max_font_size = max_font_size,

                         random_state = 42,

                         width = 800,

                         height = 400,

                         mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize = figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask)

        plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation='bilinear');

        plt.title(title, fontdict = {'size': title_size, 

                                    'verticalalignment':'bottom'})

    else:

        plt.imshow(wordcloud)

        plt.title(title, fontdict= {'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train_df["question_text"], title="Word Cloud of Questions")
## word cloud for sincere questions

train0_df = train_df[train_df['target']==0]

plot_wordcloud(train0_df["question_text"], title="Word Cloud of Questions")
## word cloud for nonsincere questions

train1_df = train_df[train_df['target']==1]

plot_wordcloud(train1_df["question_text"], title="Word Cloud of Questions")
from collections import defaultdict

train1_df = train_df[train_df["target"]==1]

train0_df = train_df[train_df["target"]==0]

# ngram generation

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



#custom function for horizontal bar chart

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



# Get barchart from sincere questions

freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Get barchart from insincere questions

freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Creating two subplots



fig = tools.make_subplots(rows = 1, cols = 2, vertical_spacing = 0.04,

                         subplot_titles = ['frequent words in sincere questions',

                                              'frequent words in insincere questions']

                         )

fig.append_trace(trace0,1,1)

fig.append_trace(trace1, 1,2)

fig['layout'].update(height =1200, width = 900, paper_bgcolor = 'rgb(233,233,233)', 

                    title ='Word count plots')

py.iplot(fig, filename = 'word_count_plots')



# frequent bigram plots in both classes



from collections import defaultdict



# ngram generation

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



#custom function for horizontal bar chart

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



# Get barchart from sincere questions

freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent,n_gram = 2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'red')



# Get barchart from insincere questions

freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent,n_gram =2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'red')



## Creating two subplots



fig = tools.make_subplots(rows = 1, cols = 2, vertical_spacing = 0.04,

                         subplot_titles = ['frequent bigrams in sincere questions',

                                              'frequent bigrams in insincere questions']

                         )

fig.append_trace(trace0,1,1)

fig.append_trace(trace1, 1,2)

fig['layout'].update(height =1200, width = 900, paper_bgcolor = 'rgb(233,233,233)', 

                    title ='Bigram count plots')

py.iplot(fig, filename = 'Bigram_count_plots')



from collections import defaultdict



# ngram generation

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



#custom function for horizontal bar chart

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



# Get barchart from sincere questions

freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent,n_gram = 3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Get barchart from insincere questions

freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent, n_gram = 3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Creating two subplots



fig = tools.make_subplots(rows = 1, cols = 2, vertical_spacing = 0.04,

                         subplot_titles = ['frequent trigrams in sincere questions',

                                              'frequent trigrams in insincere questions']

                         )

fig.append_trace(trace0,1,1)

fig.append_trace(trace1, 1,2)

fig['layout'].update(height =1200, width = 900, paper_bgcolor = 'rgb(233,233,233)', 

                    title ='Trigram count plots')

py.iplot(fig, filename = 'Trigram_count_plots')



## Number of words in text



train_df['num_words'] = train_df["question_text"].apply(lambda x : len(str(x).split(" ")))

test_df['num_words'] = test_df['question_text'].apply(lambda x: len(str(x).split(" ")))
## Number of unique words

train_df['num_unique_words'] = train_df["question_text"].apply(lambda x : len(set(str(x).split(" "))))

test_df['num_unique_words'] = test_df['question_text'].apply(lambda x: len(set(str(x).split(" "))))
# Number of characters in text

train_df['num_chars'] = train_df['question_text'].apply(lambda x : len(str(x)))

test_df['num_chars'] = test_df['question_text'].apply(lambda x: len(str(x)))
## Number of stopwords in text

train_df['num_stopwords'] = train_df['question_text'].apply(lambda x : len([w for w in str(x).lower().split(" ") if w in STOPWORDS ]))

test_df['num_stopwords'] = test_df['question_text'].apply(lambda x : len([w for w in str(x).lower().split(" ") if w in STOPWORDS ]))
train_df.head(5)
## Number of punctuations



train_df['num_punct'] = train_df['question_text'].apply(lambda x : len([w for w in str(x) if w in string.punctuation ]))

test_df['num_punct'] = test_df['question_text'].apply(lambda x : len([w for w in str(x) if w in string.punctuation ]))
## number of upper case words



train_df['num_words_upper'] = train_df['question_text'].apply(lambda x : len([w for w in str(x).split(" ") if w.isupper() ]))

test_df['num_words_upper'] = test_df['question_text'].apply(lambda x : len([w for w in str(x).split(" ") if w.isupper() ]))
## Number of title case

train_df['num_words_title'] = train_df['question_text'].apply(lambda x : len([w for w in str(x).split(" ") if w.istitle() ]))

test_df['num_words_upper'] = test_df['question_text'].apply(lambda x : len([w for w in str(x).split(" ") if w.isupper() ]))
## Average length of words in text

train_df['mean_word_len'] = train_df['question_text'].apply(lambda x : np.mean([len(w) for w in str(x).split(" ")]))

test_df['mean_word_len'] = test_df['question_text'].apply(lambda x : np.mean([len(w) for w in str(x).split(" ")]))
# Distribution of meta features in sincere and insincere questions



train_df["num_words"].loc[train_df["num_words"]>60] = 60 # truncating for better visuals

train_df['num_punct'].loc[train_df['num_punct']>10] = 10

train_df['num_chars'].loc[train_df['num_chars']>350] = 350
f, axes = plt.subplots(3,1,figsize =(10,20))



sns.boxplot(x = 'target', y ='num_words', data = train_df, ax = axes[0])

axes[0].set_xlabel('Target', fontsize = 12)

axes[0].set_title('Words count in each class', fontsize = 15)



sns.boxplot(x ='target', y='num_chars',data = train_df, ax = axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title('Characters count in each class', fontsize=15)



sns.boxplot(x = 'target', y='num_punct', data = train_df, ax = axes[2])

axes[2].set_xlabel('Target', fontsize = 12)

axes[2].set_title('Punctuations in each class', fontsize = 15)



plt.show()
## Baseline model



tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range = (1,3))

tfidf_vec.fit_transform(train_df["question_text"].values.tolist() + 

                       test_df["question_text"].values.tolist())



train_tfidf = tfidf_vec.transform(train_df["question_text"].values.tolist())

test_tfidf = tfidf_vec.transform(test_df["question_text"].values.tolist())
train_y = train_df['target'].values



def runModel(train_X, train_y, test_X, test_y, test_X2):

    model = linear_model.LogisticRegression(C=5, solver = 'sag')

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)[:,1]

    pred_test_y2 = model.predict_proba(test_X2)[:,1]

    return pred_test_y, pred_test_y2, model



print('Building model')

cv_scores =[]

pred_full_test = 0

pred_train = np.zeros(train_df.shape[0])

kf = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 2017)

for dev_index, val_index in kf.split(train_df):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    break

    
## Getting best threshold on validation sample



for thresh in np.arange(0.1, 0.201, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at thresh {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
##Lets look at important words used in the classification using eli5



eli5.show_weights(model, vec = tfidf_vec, feature_filter = lambda x: x!='<BIAS>')
train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_df.head()
## splitting training and test set

train_df, val_df = model_selection.train_test_split(train_df, test_size = 0.1, random_state = 2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## filling na values

train_X = train_df['question_text'].fillna("_na_").values

val_X = val_df['question_text'].fillna("_na_").values

test_X = test_df['question_text'].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## pad sequences

train_X = pad_sequences(train_X, maxlen= maxlen)

val_X = pad_sequences(val_X, maxlen = maxlen)

test_X = pad_sequences(test_X, maxlen = maxlen)



## get the target vales

train_y = train_df['target'].values

val_y = val_df['target'].values



inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
# Train the model

model.fit(train_X,train_y, batch_size = 512, epochs = 2, validation_data =(val_X, val_y))
pred_noemb_val_y = model.predict([val_X], batch_size = 1024, verbose = 1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at thresh {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
# testset prediction

pred_noemb_test_y = model.predict([test_X], batch_size = 1024, verbose =1)
#cleaning up memory



del model, inp, x

import gc

gc.collect()

time.sleep(10)

# using pretrained embeddings

!ls ../input/quora-insincere-questions-classification/embeddings/
# Glove Embeddings

EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())





model.fit(train_X, train_y, batch_size = 512, epochs = 2, validation_data = (val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size = 1024, verbose = 1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at thresh {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
pred_glove_test_y = model.predict(test_X, batch_size = 1024, verbose = 1)

del word_index, embedding_matrix, all_embs, embeddings_index, model, inp, x

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype ='float32')



embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean, emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape = (maxlen,))

x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)

x = Bidirectional(LSTM(64, return_sequences = True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation = 'relu')(x)

x = Dropout(0.1)(x)

x = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs= inp, outputs = x)

model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics =['accuracy'])



print(model.summary())
model.fit(train_X, train_y, batch_size = 512, epochs = 2, validation_data =(val_X, val_y))
pred_fasttext_val_y = model.predict([val_X], batch_size = 1024, verbose = 2)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.arange(thresh, 2)

    print("F1 score at thresh {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))

    
pred_fasttext_test_y = model.predict(test_X, batch_size = 1024, verbose =1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
EMBEDDING_FILE = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)



all_embs = np.stack(embeddings_index.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(LSTM(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))))
pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x

import gc; gc.collect()

time.sleep(10)
pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y 

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y

pred_test_y = (pred_test_y>0.35).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)