# all the standard imports + spacy
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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers



import matplotlib.pylab as plt
#we will import spacy and then disable the additional modules we dont need because they take a lot of compute time
import spacy
nlp = spacy.load('en', disable = ["parser", "ner", "textcat", "tagger"])
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
tokens = []
for doc in tqdm(nlp.pipe(train_df["question_text"].values, n_threads = 16)):
    tokens.append(" ".join([n.text for n in doc]))
train_df["question_text"] = tokens
results = set()
train_df['question_text'].str.lower().str.split().apply(results.update)
print("Number of unique words before pos tagging:", len(results))
print("What the text looks like before tagging:",  train_df["question_text"][0])
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
nlp = spacy.load('en', disable = ["parser", "ner", "textcat"])
tokens = []
for doc in tqdm(nlp.pipe(train_df["question_text"].values, n_threads = 16)):
    tokens.append(" ".join([n.text + "_"  + n.pos_ for n in doc]))
train_df["question_text"] = tokens
print("after tagging:",  train_df["question_text"][0])
results = set()
train_df['question_text'].str.lower().str.split().apply(results.update)
print("Number of unique words after pos tagging:", len(results))
#number of words we have added to our vocabulary
284281 - 219231
tokens = []
pos = []
for doc in tqdm(nlp.pipe(test_df["question_text"].values, n_threads = 16)):
    tokens.append([n.text + "_"  + n.pos_ for n in doc])
test_df["tokens"] = tokens
# Cross validation - create training and testing dataset
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)
# Preprocess the data
## some config values                                                                                                                                                                                       
embed_size = 300 # how big is each word vector                                                                                                                                                              
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)                                                                                                                      
maxlen = 20 # max number of words in a question to use                                                                                                                                                     

## fill up the missing values                                                                                                                                                                               
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences                                                                                                                                                                                   
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences                                                                                                                                                                                        
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values                                                                                                                                                                                    
train_y = train_df['target'].values
val_y = val_df['target'].values
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
vocab_dict = {}
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= nb_words: continue
    word_part = word.split("_")[0]
    embedding_vector = embeddings_index.get(word_part)
    if embedding_vector is not None: 
        if word_part in vocab_dict:
            vocab_dict[word_part].append((word, i))
        else:
            vocab_dict[word_part] = [(word, i)]
        embedding_matrix[i] = embedding_vector
    
#filter for words that with more than one part of speech
vocab_dict = {i:vocab_dict[i] for i in vocab_dict if len(vocab_dict[i]) > 1}
new_vocab_indexes = []
new_word_list = []
for word in vocab_dict.values():
    for pos in word:
        new_vocab_indexes.append(pos[1])
        new_word_list.append(pos[0])
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=3)
# tsne = TSNE(n_components=3)
X_embedded = tsvd.fit_transform(embedding_matrix[new_vocab_indexes])
# X_embedded = tsne.fit_transform(X_embedded)
X_embedded.shape
word_count = 5000
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np
trace1 = go.Scatter3d(
    x=X_embedded[:word_count,0],
    y=X_embedded[:word_count,1],
    z=X_embedded[:word_count,2],
    mode='markers',
    text = new_word_list[:word_count],
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple-3d-scatter')
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', restore_best_weights=True)
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Flatten, Lambda, SpatialDropout1D
import keras.backend as K
# nb_filter = 32
# inp = Input(shape=(maxlen,))

# x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
# x = SpatialDropout1D(.4)(x)
# rev_x = Lambda(lambda x: K.reverse(x,axes=-1))(x)
# conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(x)
# conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                      border_mode='valid', activation='relu')(x)
# conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                      border_mode='valid', activation='relu')(x)
# conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                      border_mode='valid', activation='relu')(x)
# convs = [conv, conv1, conv2, conv3]
# convs2 = []
# for layer in convs:
#     conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(layer)
#     conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                          border_mode='valid', activation='relu')(layer)
#     conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                          border_mode='valid', activation='relu')(layer)
#     conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                          border_mode='valid', activation='relu')(layer)
#     conv = MaxPooling1D()(conv)
#     conv1 = MaxPooling1D()(conv1)
#     conv2 = MaxPooling1D()(conv2)
#     conv3 = MaxPooling1D()(conv3)
#     convs2.append(concatenate([conv, conv1, conv2, conv3], axis = 1))

    
# conv4 = concatenate(convs2, axis = 1)


# rev_conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                      border_mode='valid', activation='relu')(rev_x)
# rev_convs = [rev_conv, rev_conv1, rev_conv2, rev_conv3]
# rev_convs2 = []
# for layer in rev_convs:
#     rev_conv = Convolution1D(nb_filter=nb_filter, filter_length=1,
#                      border_mode='valid', activation='relu')(layer)
#     rev_conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv2 = Convolution1D(nb_filter=nb_filter, filter_length=3,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv3 = Convolution1D(nb_filter=nb_filter, filter_length=4,
#                          border_mode='valid', activation='relu')(layer)
#     rev_conv = MaxPooling1D()(rev_conv)
#     rev_conv1 = MaxPooling1D()(rev_conv1)
#     rev_conv2 = MaxPooling1D()(rev_conv2)
#     rev_conv3 = MaxPooling1D()(rev_conv3)
    
#     rev_convs2.append(concatenate([rev_conv, rev_conv1, rev_conv2, rev_conv3], axis = 1))

    
# rev_conv4 = concatenate(rev_convs2, axis = 1)
# conv4 = concatenate([rev_conv4, conv4], axis = 1)

# conv5 = Flatten()(conv4)

# z = Dropout(0.5)(Dense(64, activation='relu')(conv5))
# z = Dropout(0.5)(Dense(64, activation='relu')(z))

# pred = Dense(1, activation='sigmoid', name='output')(z)

# model = Model(inputs=inp, outputs=pred)

# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['accuracy'])


# from keras.utils.vis_utils import model_to_dot
# from IPython.display import Image

# Image(model_to_dot(model, show_shapes=True).create(prog='dot', format='png'))
#the model we will actually use to measure performance and analyze errors on
inp = Input(shape=(maxlen,))
x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable = False)(inp)
x = SpatialDropout1D(.4)(x)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=3000, epochs=8, validation_data=(val_X, val_y), callbacks = [es])
for layer in model.layers:
    layer.trainable = False
    if "embedding" in layer.name:
        layer.trainable = True
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=3000, epochs=8, validation_data=(val_X, val_y), callbacks = [es])
for layer in model.layers:
    if "embedding" in layer.name:
        new_embed = layer.get_weights()[0]

X_embedded = tsvd.transform(new_embed[new_vocab_indexes])
# X_embedded = tsne.fit_transform(X_embedded)
X_embedded.shape

from sklearn.metrics.pairwise import cosine_similarity
def sum_cos_sim(embedding_matrix):
    tot_sim = 0
    sim_dict = {}
    for word, word_parts in vocab_dict.items():
        cos_sim = 0
        first_embed = embedding_matrix[word_parts[0][1]].reshape(1, -1)
        for word_part in word_parts[1:]:
            next_embed = embedding_matrix[word_part[1]].reshape(1, -1)
            cos_sim += cosine_similarity(first_embed, next_embed)/(len(word_parts) -1)
        sim_dict[word] = cos_sim
        tot_sim += cos_sim/len(vocab_dict.items())
    return tot_sim, sim_dict

sim_val, sim_dict = sum_cos_sim(embedding_matrix)
print("average cosine similarity of different word senses:" , sim_val)
sim_val, sim_dict = sum_cos_sim(new_embed)
print("average cosine similarity of different word senses after embedding tuning:" , sim_val)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import numpy as np
trace1 = go.Scatter3d(
    x=X_embedded[:word_count,0],
    y=X_embedded[:word_count,1],
    z=X_embedded[:word_count,2],
    mode='markers',
    text = new_word_list[:word_count],
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple-3d-scatter')
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

thresholds = np.arange(0.1, 0.501, 0.01)
f1s = np.zeros(thresholds.shape[0])

for ind, thresh in np.ndenumerate(thresholds):
    f1s[ind[0]] = metrics.f1_score(val_y, (pred_noemb_val_y > np.round(thresh, 2)).astype(int))

np.round(thresholds[np.argmax(f1s)], 2)
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # else:
    # print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
pred_noemb_val_y[:, 0].shape
pred_noemb_val_y1 = pred_noemb_val_y[:, 0]
y_test = val_y
opt_thresh = np.round(thresholds[np.argmax(f1s)], 2)
# y_test = val_y
y_pred = (pred_noemb_val_y > opt_thresh).astype(int)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

opt_thresh = np.round(thresholds[np.argmax(f1s)], 2)
y_test = val_y
y_pred = (pred_noemb_val_y > opt_thresh).astype(int)

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Sincere','Insincere'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
precision = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[0,1])
recall = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0])
print("Precision: " + str(np.round(precision, 3)))
print("Recall: " + str(np.round(recall, 3)))
"F1 Score:", (2 * precision * recall)/(precision + recall)
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
original_text = val_df["question_text"].fillna("_na_").values
import operator
from tqdm import tqdm
def analyze_model(model, num_results, reverse = False):
    #let's see on which comments we get the biggest loss
    train_predictions = model.predict([val_X], batch_size=250, verbose=1)
    inverted_word_index = dict([[v,k] for k,v in word_index.items()])
    pd.DataFrame(train_predictions).hist()
    results = []
    eps = 0.1 ** 64
    for i in tqdm(range(0, len(val_y))):
        metric = 0

        for j in range(len([val_y[i]])):
            p = train_predictions[i][j]
            y = [val_y[i]][j]
            metric +=  -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        if p < opt_thresh and y == 1:
            results.append((original_text[i], metric, val_y[i], train_predictions[i], val_X[i]))
    results.sort(key=operator.itemgetter(1), reverse=reverse)  
    
    for i in range(num_results):
        inverted_text = ""
        for index in results[i][4]:
            if index > 0:
                word = inverted_word_index[index]
                if not np.any(embedding_matrix[index]):
                    word = "_" + word + "_"
                inverted_text += word + " "


        print(str(results[i][2]) + "\t" + str(results[i][3]) + "\t" + str(results[i][1]))
        print("Original Text")
        print( str(results[i][0]))
        print("---------------------------")
        print("Text that reached the model")
        print(inverted_text)
        print("")
#500 highest loss comments
#Correct Label | Model Output | Loss
#Original Text
#===========
#Text that reached the model after preprocessing, tokenizing and embedding
analyze_model(model, 500, False)

