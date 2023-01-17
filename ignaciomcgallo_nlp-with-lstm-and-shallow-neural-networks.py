import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator



#from string import punctuation



from sklearn.model_selection import train_test_split

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")



# NLP libraires



import gensim # pip install gensim

from gensim.models.word2vec import Word2Vec # word2vec model gensim class

TaggedDocument = gensim.models.doc2vec.TaggedDocument

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk



%matplotlib inline
df_train = pd.read_csv("../input/imdb-dataset-sentiment-analysis-in-csv-format/Train.csv", encoding='UTF-8')
df_train.head()
df_train.shape
df_test = pd.read_csv("../input/imdb-dataset-sentiment-analysis-in-csv-format/Test.csv", encoding='UTF-8')
df_test.shape
data = pd.concat([df_train,df_test])

data.head()
data.shape
data.describe()
data.label.isnull().sum()
data = pd.concat([df_train,df_test])
data.head()
def tokenizer(text):

    try:



        text  = text.lower()

        tokens = TweetTokenizer().tokenize(text)

        

        return tokens

    except:

        return 'NC'
def process(data):

    # progress_map is a variant of the map function plus a progress bar.

    # Handy to monitor DataFrame creations:

    data['tokens'] = data['text'].progress_map(tokenizer)

    data = data[data.tokens != 'NC']

    data.reset_index(inplace=True)

    data.drop('index', inplace=True, axis=1)

    return data



data = process(data)
data.head()
tokenized_train = data["tokens"][:len(df_train)].values

tokenized_test = data["tokens"][len(df_train):].values

y_train = data["label"][:len(df_train)].values

y_test = data["label"][len(df_train):].values
print(tokenized_train[0])

print(y_train[0])
vec_dim = 40





# Model word2vec:

text_w2v = Word2Vec(size=vec_dim, min_count=5) 



# Building the vocabulary:

text_w2v.build_vocab(tokenized_train)



# Training the model:

text_w2v.train(tokenized_train, total_examples=text_w2v.corpus_count, epochs=text_w2v.epochs)
text_w2v.save("text_w2v.h5")
text_w2v.wv['good']
text_w2v.wv.most_similar('good')
text_w2v.wv.most_similar('mountain')
text_w2v.wv.most_similar('actor')
text_w2v.wv.most_similar('amazing')
# importing bokeh library for interactive data visualization

import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook



# defining the chart

output_notebook()

fig = bp.figure(plot_width=700, plot_height=600, title="Map of word vectors",

                tools="pan,wheel_zoom,box_zoom,reset,hover,save",

                x_axis_type=None, y_axis_type=None, min_border=1)



# getting a list of word vectors. limit to 2500. each is of 40 dimensions

word_vectors = [text_w2v.wv[w] for w in list(text_w2v.wv.vocab.keys())[:2500]]



# dimensionality reduction. converting the vectors to 2d vectors

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, early_exaggeration=10, random_state=0, init='pca')

tsne_w2v = tsne_model.fit_transform(word_vectors)



# putting everything in a dataframe

tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])

tsne_df['words'] = list(text_w2v.wv.vocab.keys())[:2500]



# plotting. the corresponding word appears when you hover on the data point.

fig.scatter(x='x', y='y', source=tsne_df)

hover = fig.select(dict(type=HoverTool))

hover.tooltips={"word": "@words"}

show(fig)
len(text_w2v.wv.vocab)
print('building tf-idf matrix ...')

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=5)

vectorizer.fit(tokenized_train)

IDFs = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print('size of vocabulary obtained with TfidfVectorizer:', len(IDFs))

print('size of vocabulary obtained with word2vec:', len(text_w2v.wv.vocab))

print("Some idfs:")

aux = list(IDFs.items())

for i in list(range(3))+list(range(1000,1005)):

    print("  ", aux[i])
IDFs
import pickle



with open("IDFs.pkl", "wb") as f:

    pickle.dump(IDFs, f)
def Text2Vec(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += text_w2v.wv[word].reshape((1, size)) * IDFs[word]

            count += 1.

        except KeyError: # handling the case where the token is not

                         # in the corpus. useful for testing.

            continue

    if count != 0:

        vec /= count

    return vec
text_vecs_train = np.zeros((len(tokenized_train), vec_dim ))

for i,x in tqdm(enumerate(tokenized_train)):

    text_vecs_train[i] = Text2Vec(x, vec_dim)
text_vecs_test = np.zeros((len(tokenized_test), vec_dim))

for i,x in tqdm(enumerate(tokenized_test)):

    text_vecs_test[i] = Text2Vec(x, vec_dim)
print(text_vecs_train.shape)

print(text_vecs_test.shape)
from sklearn.preprocessing import StandardScaler, normalize



scaler = StandardScaler()

scaler.fit(text_vecs_train)

text_vecs_train_sc = scaler.transform(text_vecs_train)

text_vecs_test_sc  = scaler.transform(text_vecs_test)

from sklearn.decomposition import PCA



pca = PCA()

X_pca_train = pca.fit_transform(text_vecs_train_sc)
#import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool, LabelSet, ColumnDataSource, Range1d



#from bokeh.plotting import figure, show, output_notebook



pc_x = 0

pc_y = 1



n_visualizar_por_clase = 5000



pcs_names = ["main component "+str(i+1) for i in range(vec_dim)]



colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

markers = ['o', 's']



# defining the chart

output_notebook()

p = bp.figure(plot_width=700, plot_height=600, title="Text vecs, PCA space",

              tools="pan,wheel_zoom,box_zoom,reset,hover,save",

              x_axis_label=pcs_names[pc_x],

              y_axis_label=pcs_names[pc_y],              

              #x_axis_type=None, y_axis_type=None,

              min_border=1)

p.title.text_font_size = '16pt'

p.xaxis.axis_label_text_font_style='normal'

p.xaxis.axis_label_text_font_size='16pt'

p.yaxis.axis_label_text_font_style='normal'

p.yaxis.axis_label_text_font_size='16pt'



p.xgrid.visible = False

p.ygrid.visible = False



for label,color,marker in zip(np.unique(y_train),colors,markers):

    inds = np.where(y_train==label)[0][:n_visualizar_por_clase]

    dictf = {'x':X_pca_train[inds,pc_x],

             'y':X_pca_train[inds,pc_y],

             'Class':len(inds)*[label],

             'Text':[" ".join(a) for a in tokenized_train[inds]],

             'row':inds}

    p.scatter(x='x', y='y', source=ColumnDataSource(dictf), color=color,

              legend='Class {}'.format(label), alpha=0.1)

    hover = p.select(dict(type=HoverTool))

    

    hover.tooltips={"Class":"@Class",

                    "Text":"@Text",

                    "row":"@row"}

show(p)
def training_graph(tr_acc, val_acc):

    ax=plt.figure(figsize=(10,4)).gca()

    plt.plot(1+np.arange(len(tr_acc)), 100*np.array(tr_acc))

    plt.plot(1+np.arange(len(val_acc)), 100*np.array(val_acc))

    plt.title('Model hit rate (%)', fontsize=18)

    plt.ylabel('Hit rate (%)', fontsize=18)

    plt.xlabel('Epoch', fontsize=18)

    plt.legend(['Training', 'Validation'], loc='upper left')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
from keras.models import Sequential

from keras.layers import Dense



model1 = Sequential()

model1.add(Dense(32, activation='relu', input_dim=vec_dim))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop',

               loss='binary_crossentropy',

               metrics=['accuracy'])
from keras.models import load_model

from keras.callbacks import ModelCheckpoint



BATCH_SIZE = 32

nepochs = 15

TRAIN1 = True

filepath1 = "best_model1.h5"





if TRAIN1:

    acum_tr_acc = []

    acum_val_acc = []

    checkpoint = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1,

                                 save_best_only=True,

                                 mode='max') 

    callbacks_list = [checkpoint]

    

    for i in range(nepochs):

        history = model1.fit(text_vecs_train_sc, y_train,

                             batch_size=BATCH_SIZE,

                             epochs=1,

                             verbose=2,

                             callbacks=callbacks_list,

                             validation_split=0.3,

                             shuffle=False,

                            )

    

        acum_tr_acc = acum_tr_acc + history.history['accuracy']

        acum_val_acc = acum_val_acc + history.history['val_accuracy']

        if len(acum_tr_acc) > 1:

            training_graph(acum_tr_acc, acum_val_acc)



model1 = load_model(filepath1)

y_train_pred_prob1 = model1.predict(text_vecs_train_sc)

y_test_pred_prob1 = model1.predict(text_vecs_test_sc)



y_train_pred1 = y_train_pred_prob1.round()

y_test_pred1  = y_test_pred_prob1.round()
from sklearn.metrics import accuracy_score



print("Accuracy (training): %.2f%%" % (100*accuracy_score(y_train, y_train_pred1)))

print("Accuracy (test): %.2f%%"     % (100*accuracy_score(y_test,  y_test_pred1)))
index2word = list(text_w2v.wv.vocab)

embedding_matrix = text_w2v.wv[index2word]



padding_vector = np.zeros((1, vec_dim))

index2word.insert(0,'()')

index2word = np.array(index2word)



word2index = dict(zip(index2word, range(len(index2word))))



embedding_matrix = np.vstack((padding_vector, embedding_matrix))



print(np.shape(index2word))

print(np.shape(embedding_matrix))
index2word[253]
word2index["square"]
print(tokenized_train[30][:5])



word2index[tokenized_train[30][4]]
def code_word(y):

    try:

        return word2index[y]

    except KeyError:

        return



X_train_coded = []

for text in tokenized_train:

    X_train_coded.append([y for y in [code_word(x) for x in text] if y != None])



X_test_coded = []

for text in tokenized_test:

    X_test_coded.append([y for y in [code_word(x) for x in text] if y != None])
print(tokenized_train[1])

X_train_coded[1]
index2word[X_train_coded[1]]
aux = [len(x) for x in X_train_coded]

i = np.argmax(aux)

text_mas_palabras = X_train_coded[i]

print("Maximum number of words in text:", len(text_mas_palabras))

print("Text:", tokenized_train[i])

print("Text (word codes):", text_mas_palabras)
plt.figure(figsize=(20,10))

f = sns.countplot(aux)

plt.axis([0,400,0,300])

plt.title("Histogram: number of words", fontsize=18);
max_words_text = 400



from keras.preprocessing.sequence import pad_sequences



X_train_pad = pad_sequences(X_train_coded, maxlen=max_words_text)

X_test_pad  = pad_sequences(X_test_coded,  maxlen=max_words_text)
X_train_pad[:3]
embedding_matrix.shape
from keras.layers.embeddings import Embedding

from keras.layers import LSTM



embedding_layer = Embedding(embedding_matrix.shape[0],

                            embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=max_words_text,

                            trainable=False)
set_dropout=False



# create the model

model2 = Sequential()

#model2.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length,

#                    activity_regularizer='l2'))

model2.add(embedding_layer)

if set_dropout:

    model2.add(Dropout(0.2))

#model2.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))

model2.add(LSTM(20, return_sequences=True))

model2.add(LSTM(20))

if set_dropout:

    model2.add(Dropout(0.2))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy']) #    'adam'
print(model2.summary())
print(X_train_pad.shape)

print(X_test_pad.shape)
from keras.models import load_model



BATCH_SIZE = 256

nepochs = 5

#TRAIN2 = False

TRAIN2 = True

filepath2 = "best_model2.h5"





if TRAIN2:

    acum_tr_acc = []

    acum_val_acc = []

    checkpoint = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1,

                                 save_best_only=True,

                                 mode='max') 

    callbacks_list = [checkpoint]

    

    for i in range(nepochs):

        history = model2.fit(X_train_pad, y_train,

                             batch_size=BATCH_SIZE,

                             epochs=1,

                             verbose=1,

                             callbacks=callbacks_list,

                             validation_split=0.3,

                             shuffle=False,

                            )

    

        acum_tr_acc = acum_tr_acc + history.history['accuracy']

        acum_val_acc = acum_val_acc + history.history['val_accuracy']

        if len(acum_tr_acc) > 1:

            training_graph(acum_tr_acc, acum_val_acc)



model2 = load_model(filepath2)

y_train_pred_prob2 = model2.predict(X_train_pad)

y_test_pred_prob2  = model2.predict(X_test_pad)



y_train_pred2 = y_train_pred_prob2.round()

y_test_pred2  = y_test_pred_prob2.round()
score_train2 = accuracy_score(y_train, y_train_pred2)

score_test2  = accuracy_score(y_test,  y_test_pred2)



print("Accuracy (training): %.2f%%" % (100*score_train2))

print("Accuracy (test)    : %.2f%%" % (100*score_test2))