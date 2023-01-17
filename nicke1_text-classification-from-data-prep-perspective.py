#set seed

seed = 1029



#import data

import pandas as pd

train = pd.read_csv('../input/train.csv')



#divide data info train (90%), test (5%) and valid (5%) set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train[['question_text']], train[['target']], 

                                                    test_size=0.2, random_state=seed,

                                                    stratify=train['target'].tolist(),

                                                    shuffle = True)

X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=seed,

                                                    stratify=y_test,shuffle = True)
train.head(5)
#clean up

import gc 

import time 



del(train)

gc.collect()

time.sleep(2)
import matplotlib.pyplot as plt #for vizualization of data

from pylab import rcParams

import numpy as np

from collections import Counter

%matplotlib inline



cmap = plt.get_cmap("tab20c")

colors = cmap((np.arange(10)))



rcParams['figure.figsize'] = 20, 5

fig, ax = plt.subplots(1, 4, sharex='col', sharey='row')



ax[0].pie([X_train.shape[0],X_test.shape[0],X_valid.shape[0]], explode=(0, 0.1,0.1), 

          labels= ["train \n{}mln".format(round(X_train.shape[0]/1000000,2)), "test\n{}K".format(round(X_test.shape[0]/1000,1)),"valid\n{}K".format(round(X_valid.shape[0]/1000,1))], autopct='%1.0f%%',

        shadow=True, startangle=75, colors=colors)

ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[0].set_title('Data splits \n', fontsize=20)



ax[1].pie(Counter(y_train.target).values(), explode=(0,0.1), labels= ["sincere","insincere"], autopct='%1.0f%%',

        shadow=True, startangle=45, colors=colors)

ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1].set_title('Train set \n', fontsize=20)



ax[2].pie(list(Counter(y_test.target).values())[::-1], explode=(0,0.1), labels= ["sincere","insincere"], autopct='%1.0f%%',

        shadow=True, startangle=45, colors=colors)

ax[2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[2].set_title('Test set \n', fontsize=20)



ax[3].pie(Counter(y_valid.target).values(), explode=(0,0.1), labels= ["sincere","insincere"], autopct='%1.0f%%',

        shadow=True, startangle=45, colors=colors)

ax[3].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[3].set_title('Valid set \n', fontsize=20)

plt.show()
import time

start = time.time()

import numpy as np

#y_pred = list(np.repeat(0.99,len(y_valid)))

np.random.seed(1029)

y_pred = list(np.random.uniform(size=len(y_valid)))

stop = time.time()

time_elapsed = stop-start
from sklearn.metrics import f1_score, accuracy_score



def bestThressholdF1(y_train_,train_preds_):

    tmp = [0,0,0] # idx, cur, max

    delta = 0

    for tmp[0] in np.arange(0.1, 0.501, 0.01):

        tmp[1] = f1_score(y_train_, np.array(train_preds_)>tmp[0])

        if tmp[1] > tmp[2]:

            delta = tmp[0]

            tmp[2] = tmp[1]

    return tmp[2]



def bestThressholdACC(y_train_,train_preds_):

    tmp = [0,0,0] # idx, cur, max

    delta = 0

    for tmp[0] in np.arange(0.1, 0.501, 0.01):

        tmp[1] = accuracy_score(y_train_, np.array(train_preds_)>tmp[0])

        if tmp[1] > tmp[2]:

            delta = tmp[0]

            tmp[2] = tmp[1]

    return tmp[2]



from sklearn import metrics



def get_scores(y_train__,train_preds__):

    dict_ = {}

    dict_['F1']=bestThressholdF1(y_train__,train_preds__)

    dict_['accuracy'] = bestThressholdACC(y_train__,train_preds__)

    fpr, tpr, thresholds = metrics.roc_curve(y_train__,train_preds__, pos_label=1)

    dict_['auc']=metrics.auc(fpr, tpr)

    dict_['gini'] = (metrics.auc(fpr, tpr)-0.5)*2

    return dict_
scores = []

scores.append(('Random guessing',get_scores(y_valid.target,y_pred), y_pred, 

               time.strftime("%Mmin %Ssec", time.gmtime(time_elapsed)),

              0))
scores[-1][1]
import gc 

import time 



del(y_pred)

gc.collect()

time.sleep(2)
def text_length(_list):

    return list(map(lambda x: len(x),_list))



def text_words(_list):

    return list(map(lambda x: len(x.split()),_list))



def text_density(_list):

    return np.array(text_words(_list))/np.array(text_length(_list))



def text_title_words(_list):

    return list(map(lambda x: sum([w.istitle() for w in x.split()]),_list))



def text_capital_words(_list):

    return list(map(lambda x: sum(1 for c in x if c.isupper()),_list))



def text_caps_vs_length(_list):

    return list(map(lambda x: sum(1 for c in x if c.isupper())/len(x),_list))



def text_unique(_list):

    return list(map(lambda x: len(set(w for w in x.split())),_list))



def text_words_vs_unique(_list):

    return list(map(lambda x: len(set(w for w in x.split()))/len(x.split()),_list))



from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))



def text_stopwords(_list):

    return list(map(lambda x: sum([1 if i in stopWords else 0 for i in x.split()]),_list))
from sklearn.base import BaseEstimator, TransformerMixin



class ApplyFunctionCT(BaseEstimator, TransformerMixin):

    

    def __init__(self, function, **kwargs):

        self.function = function

        self.kwargs = kwargs

        

    def fit(self, x, y = None):

        return self

    

    def get_feature_names(self):

        if hasattr(self, "columnNames"):

            return self.columnNames

        else:

            return None  

    

    def transform(self, x):

        if len(self.kwargs) == 0:

            wyn = x.apply(self.function)

        else:

            wyn = x.apply(self.function, param = self.kwargs)

        self.columnNames = wyn.columns

        return wyn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer,RobustScaler, MaxAbsScaler

from sklearn.pipeline import make_pipeline, make_union

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

import category_encoders as ce



pipe = make_pipeline(

        make_pipeline(

            ColumnTransformer([

                ('text_length', ApplyFunctionCT(function=text_length),['question_text']),

                ('text_words', ApplyFunctionCT(function=text_words),['question_text']),

                ('text_density', ApplyFunctionCT(function=text_density),['question_text']),

                ('text_title_words', ApplyFunctionCT(function=text_title_words),['question_text']),

                ('text_capital_words', ApplyFunctionCT(function=text_capital_words),['question_text']),

                ('text_caps_vs_length', ApplyFunctionCT(function=text_caps_vs_length),['question_text']),

                ('text_unique', ApplyFunctionCT(function=text_unique),['question_text']),

                ('text_words_vs_unique', ApplyFunctionCT(function=text_words_vs_unique),['question_text']),

                ]),

            KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile'),        

            ),

    LogisticRegression(penalty = 'l2', C = 0.2,  random_state=seed, solver = 'lbfgs',max_iter=400, 

                       verbose=1, n_jobs=-1) 

)
import time

start = time.time()



pipe.fit(X_train, y_train.target)



stop = time.time()

time_elapsed = stop-start
import warnings

warnings.simplefilter("ignore")



y_predict = pipe.predict_proba(X_valid)[:,1]

scores.append(('Manual feature engineering',get_scores(y_valid,y_predict),y_predict, 

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

              (pipe.named_steps['logisticregression'].coef_).size))



print(scores[-1][1])
import matplotlib.pyplot as plt #for vizualization of data

%matplotlib inline



from pylab import rcParams

import numpy as np

from collections import Counter

import seaborn as sns



def plot_progress(list_):

    fig, ax = plt.subplots(figsize=(14, 6))

    #sns.set_style("whitegrid")



    plt.plot(list(range(len(list_))), [i[1]['gini'] for i in list_], '-ok')

    for j,i in enumerate(zip([i[1]['gini'] for i in list_],

                 list(range(len(list_))),

                 [i[0] for i in list_],

                 [i[3] for i in list_],

                 [i[4] for i in list_])):

        ax.text(i[1], i[0]-0.08, 

                i[2]+',\nG: '+str(round(i[0],3))+', t: '+i[3],rotation=-25, size=10, color = ['black','blue'][j%2])



    ax.patch.set_facecolor('white')

    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)



    ax.spines['bottom'].set_color('0.5')

    ax.spines['top'].set_color(None)

    ax.spines['left'].set_color('0.5')

    ax.spines['right'].set_color(None)



    plt.title('Progress Tracker', fontsize=12)

    plt.xlabel('Models', fontsize=11)

    plt.ylabel('Gini', fontsize=11)

    plt.xticks(fontsize=9)

    plt.yticks(fontsize=9)

    pass

plot_progress(scores)
import gc 

import time 



del(pipe, y_predict)

gc.collect()

time.sleep(2)
max_features_Vectorizer = None
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer,RobustScaler, MaxAbsScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression



pipe = make_pipeline(

    ColumnTransformer([

        ('CV', TfidfVectorizer(lowercase=True, 

                               ngram_range=(1, 1), 

                               max_features=max_features_Vectorizer, 

                               dtype=np.float32,

                               use_idf=True),'question_text')]),

    LogisticRegression(penalty = 'l2', C = 2,  random_state=seed, solver = 'lbfgs',max_iter=400, 

                       verbose=1, n_jobs=-1)

    )
import time

start = time.time()



pipe.fit(X_train, y_train.target)



stop = time.time()

time_elapsed = stop-start
#x = pipe.named_steps['columntransformer'].transform(X_train)

#print("Sparsity equals {}- the number of zero-valued elements divided by the total number of elements".format(

#    ((x.shape[0]*x.shape[1]) -x.getnnz())/(x.shape[0]*x.shape[1])))
y_predict = pipe.predict_proba(X_valid)[:,1]

scores.append(('TF-IDF Vectorizer, Ridge Regression',get_scores(y_valid,y_predict),y_predict,

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               (pipe.named_steps['logisticregression'].coef_).size))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(pipe, y_predict)

gc.collect()

time.sleep(2)
max_features_ = 10000
from keras.models import Sequential

from keras.layers import Dense, Activation, InputLayer, BatchNormalization, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.initializers import glorot_normal



# For custom metrics

import keras.backend as K



def create_model():

    model = Sequential([

        Dense(units=192,input_dim=max_features_,kernel_initializer=glorot_normal(seed=seed)),

        BatchNormalization(),

        Activation('relu'),

        Dropout(0.2,seed=seed),

        Dense(units=64,input_dim=max_features_,kernel_initializer=glorot_normal(seed=seed)),

        BatchNormalization(),

        Activation('relu'),

        Dropout(0.2,seed=seed),

        Dense(units=32,input_dim=max_features_,kernel_initializer=glorot_normal(seed=seed)),

        BatchNormalization(),

        Activation('relu'),

        Dropout(0.2,seed=seed),

        Dense(1),

        Activation('sigmoid'),

    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
create_model().summary()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer,RobustScaler, MaxAbsScaler

from sklearn.pipeline import make_pipeline



pipe = make_pipeline(

    ColumnTransformer([

        ('CV', TfidfVectorizer(lowercase=True, ngram_range=(1, 1), max_features=max_features_, dtype=np.float32,

                               use_idf=True),'question_text')]),

    KerasClassifier(build_fn=create_model, epochs=3, batch_size=512, verbose=1)

    )
import time

start = time.time()



pipe.fit(X_train, y_train.target)



stop = time.time()

time_elapsed = stop-start
y_predict = pipe.predict_proba(X_valid)[:,1]

scores.append(('TF-IDF Vectorizer, Keras MLP',get_scores(y_valid,y_predict),y_predict,

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               pipe.named_steps['kerasclassifier'].model.count_params()))
scores[-1][1]
plot_progress(scores)
#settings 

maxlen = 70 # max number of words in a question to use

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features,

                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 

                      lower=True, 

                      split=' ', 

                      char_level=False, 

                      oov_token=None, 

                      document_count=0)
tokenizer.fit_on_texts(X_train.question_text.tolist())
word_index = tokenizer.word_index
im = 10

a = {}

for j,i in enumerate(tokenizer.word_index):

    a[i]=tokenizer.word_index[i]

    if j>=im:

        break

a
X_train_seq = tokenizer.texts_to_sequences(X_train.question_text.tolist())

X_test_seq = tokenizer.texts_to_sequences(X_test.question_text.tolist())

X_valid_seq = tokenizer.texts_to_sequences(X_valid.question_text.tolist())
print(X_train.question_text.tolist()[0])

print(X_train_seq[0])

print('------------------------------------------------------------------------')

print(X_train.question_text.tolist()[100])

print(X_train_seq[100])

print('------------------------------------------------------------------------')

print(X_train.question_text.tolist()[202])

print(X_train_seq[202])
X_train_seq = pad_sequences(X_train_seq, maxlen=maxlen)

X_test_seq = pad_sequences(X_test_seq, maxlen=maxlen)

X_valid_seq = pad_sequences(X_valid_seq, maxlen=maxlen)
print(X_train.question_text.tolist()[0])

print(X_train_seq[0])

print('------------------------------------------------------------------------')

print(X_train.question_text.tolist()[100])

print(X_train_seq[100])

print('------------------------------------------------------------------------')

print(X_train.question_text.tolist()[202])

print(X_train_seq[202])
import gc 

import time 



del(tokenizer)

gc.collect()

time.sleep(2)
def load_glove(word_index, max_features__ = max_features):

    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    

    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = -0.005838499,0.48782197

    embed_size = all_embs.shape[1]



    nb_words = min(max_features__, len(word_index))

    np.random.seed(1029)

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features__: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 
glove_embeddings = load_glove(word_index)
import seaborn as sns; sns.set()

sz = 30

plt.figure(figsize=(9,9))

plt.title('Embedding for sentence `the quick fast brown fox jumps over the lazy slow dog` ({} of 300 vecotors) \n'.format(str(sz)))

ax = sns.heatmap(pd.DataFrame(np.hstack([glove_embeddings[word_index['the'],:].reshape(-1,1),

           glove_embeddings[word_index['quick'],:].reshape(-1,1),

           glove_embeddings[word_index['fast'],:].reshape(-1,1),

           glove_embeddings[word_index['brown'],:].reshape(-1,1),

           glove_embeddings[word_index['fox'],:].reshape(-1,1),

           glove_embeddings[word_index['jumps'],:].reshape(-1,1),

           glove_embeddings[word_index['over'],:].reshape(-1,1),

           glove_embeddings[word_index['the'],:].reshape(-1,1),

           glove_embeddings[word_index['lazy'],:].reshape(-1,1),

           glove_embeddings[word_index['slow'],:].reshape(-1,1),

           glove_embeddings[word_index['dog'],:].reshape(-1,1)  

          ])[:sz,:],columns = ['the','quick','fast','brown','fox','jumps','over','the','lazy','slow','dog']

                   ,index = ['word feature {}'.format(str(i)) for i in range(sz)]

                             ), 

                 cbar=False,annot=True,annot_kws={"size": 10})
#nb_words = len(word_index)+1

nb_words = glove_embeddings.shape[0]

WV_DIM = glove_embeddings.shape[1]

maxlen = maxlen

SEED = 1029
from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=seed)(embedded_sequences)

x = SimpleRNN(64, return_sequences=False,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed))(embedded_sequences)



#output (batch, 64)

#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(x)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)

from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
model.summary()
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('RNN, Many to One',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),list(pred_val[:,0]),

              time.strftime("%M:%S", time.gmtime(time_elapsed)),

              model.count_params()

              ))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(model, hist, pred_val)

gc.collect()

time.sleep(2)

from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=SEED)(embedded_sequences)

x = CuDNNLSTM(64, return_sequences=False,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed))(embedded_sequences)



#output (batch, 64)

#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(x)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)

from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('LSTM, Many to One',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),list(pred_val[:,0]),

              time.strftime("%M:%S", time.gmtime(time_elapsed)),

              model.count_params()

              ))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(model, hist, pred_val)

gc.collect()

time.sleep(2)
from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=SEED)(embedded_sequences)

x = Bidirectional(CuDNNLSTM(64, return_sequences=False,

                  return_state = False,

                  kernel_initializer=glorot_normal(seed=seed),

                  recurrent_initializer=orthogonal(seed=seed)))(embedded_sequences)



#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(x)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)

from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('BiLSTM, Many to One',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),

               list(pred_val[:,0]),

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               model.count_params()

              ))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(model, hist, pred_val)

gc.collect()

time.sleep(2)

from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=SEED)(embedded_sequences)

x, forward_h, forward_c, backward_h, backward_c = Bidirectional(CuDNNLSTM(64, return_sequences=True,return_state = True,

                  kernel_initializer=glorot_normal(seed=seed),

                  recurrent_initializer=orthogonal(seed=seed)))(embedded_sequences)

state_h = concatenate([forward_h, backward_h])

state_c = concatenate([forward_c, backward_c])



#x (?, 70, 128)

#forward_h (?, 64)

#forward_c (?, 64)

#backward_h (?, 64)

#backward_c (?, 64)



avg_pool = GlobalAveragePooling1D()(x)

#avg_poll = (?, 128)

max_pool = GlobalMaxPooling1D()(x)

#max_pool = (?, 128)



conc = concatenate([state_h, avg_pool, max_pool])

#conc (?, 384)



#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(conc)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)

from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('BiLSTM, Many to Many, Max Pooling',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),

               list(pred_val[:,0]),

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               model.count_params()

              ))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(model, hist, pred_val)

gc.collect()

time.sleep(2)



#get_scores(y_valid,np.average(np.vstack([i[2] for i in scores[2:]]),axis=0).reshape(-1))
from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=seed)(embedded_sequences)



#Third layer 

x = Bidirectional(CuDNNLSTM(64, return_sequences=True,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed)))(embedded_sequences)



#Fourth layer 

x, x_h, x_c  = Bidirectional(CuDNNGRU(64, return_sequences=True,return_state = True,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed)))(x)



#concatenate



avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([x_h, avg_pool, max_pool])

#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(conc)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)

from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('BiLSTM+BiGRU, M2M, Max Pooling',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),

               list(pred_val[:,0]),

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               model.count_params()

              ))
scores[-1][1]
plot_progress(scores)
import gc 

import time 



del(model, hist, pred_val)

gc.collect()

time.sleep(2)



#get_scores(y_valid,np.average(np.vstack([i[2] for i in scores[2:]]),axis=0).reshape(-1))
from keras import backend as K

from keras.engine.topology import Layer

#from keras import initializations

from keras import initializers, regularizers, constraints



#https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043

class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        """

        Keras Layer that implements an Attention mechanism for temporal data.

        Supports Masking.

        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]

        # Input shape

            3D tensor with shape: `(samples, steps, features)`.

        # Output shape

            2D tensor with shape: `(samples, features)`.

        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.

        The dimensions are inferred based on the output shape of the RNN.

        Example:

            model.add(LSTM(64, return_sequences=True))

            model.add(Attention())

        """

        self.supports_masking = True

        #self.init = initializations.get('glorot_uniform')

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        # do not pass the mask to the next layers

        return None



    def call(self, x, mask=None):

        # eij = K.dot(x, self.W) TF backend doesn't support it



        # features_dim = self.W.shape[0]

        # step_dim = x._keras_shape[1]



        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        # apply mask after the exp. will be re-normalized next

        if mask is not None:

            # Cast the mask to floatX to avoid float64 upcasting in theano

            a *= K.cast(mask, K.floatx())



        # in some cases especially in the early stages of training the sum may be almost zero

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

    #print weigthted_input.shape

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        #return input_shape[0], input_shape[-1]

        return input_shape[0],  self.features_dim
from keras import Input

from keras.layers import Embedding, SpatialDropout1D, CuDNNLSTM, CuDNNGRU, Dropout, Dense, SimpleRNN

from keras.layers import concatenate, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.initializers import glorot_normal, orthogonal

# First layer

# create input

main_input = Input(shape=(maxlen,), dtype='int32',name='main_input')

# creating the embedding

embedded_sequences = Embedding(input_dim = nb_words,

                               output_dim = WV_DIM,

                               mask_zero=False,

                               weights=[glove_embeddings],

                               input_length=maxlen,

                               trainable=False)(main_input)

#Second layer

embedded_sequences = SpatialDropout1D(0.2, seed=seed)(embedded_sequences)



#Third layer 

x = Bidirectional(CuDNNLSTM(64, return_sequences=True,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed)))(embedded_sequences)



#Fourth layer 

x, x_h, x_c  = Bidirectional(CuDNNGRU(64, return_sequences=True,return_state = True,

                            kernel_initializer=glorot_normal(seed=seed),

                            recurrent_initializer=orthogonal(seed=seed)))(x)



#concatenate



att = Attention(maxlen)(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([x_h, avg_pool, max_pool, att])



#The input format should be three-dimensional: the three components represent sample size, number of time steps and output dimension

preds = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=seed))(conc)

preds = Dropout(0.1,seed=seed)(preds)

preds = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=seed))(preds)



from keras.models import Model

from keras.optimizers import Adam

model = Model(inputs=main_input, outputs=preds)

model.compile(loss='binary_crossentropy', optimizer=Adam(clipvalue=1), metrics=['accuracy'])
import time

start = time.time()



hist = model.fit(X_train_seq, y_train, batch_size=1024, epochs=5, 

                 validation_data=(X_test_seq, y_test))



stop = time.time()

time_elapsed = stop-start
pred_val = model.predict(X_valid_seq, batch_size=1024, verbose=1)
scores.append(('BiLSTM+BiGRU, M2M, Pool + Att',get_scores(y_valid.target.tolist(),list(pred_val[:,0])),

               list(pred_val[:,0]),

               time.strftime("%M:%S", time.gmtime(time_elapsed)),

               model.count_params()

              ))

scores[-1][1]
plot_progress(scores)
import time

start = time.time()

pred_val = np.average(np.vstack([i[2] for i in scores[-4:]]),axis=0).reshape(-1)



stop = time.time()

time_elapsed = stop-start
scores.append(('Blend last 4 models',get_scores(y_valid,pred_val),list(pred_val),

              time.strftime("%M:%S", time.gmtime(time_elapsed)),

              'n/a'

              ))

scores[-1][1]
plot_progress(scores)