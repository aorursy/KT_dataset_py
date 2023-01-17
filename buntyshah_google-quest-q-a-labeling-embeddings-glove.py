import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

import time

import pickle

import re

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()

from pathlib import Path

from scipy.stats import spearmanr
GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
%%time

datadir = Path('/kaggle/input/google-quest-challenge')



# Read in the data CSV files

train = pd.read_csv(datadir/'train.csv')

test = pd.read_csv(datadir/'test.csv')

sample_submission = pd.read_csv(datadir/'sample_submission.csv')
import pandas as pd

sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")

train = pd.read_csv("../input/google-quest-challenge/train.csv")
feature_columns = [col for col in train.columns if col not in sample_submission.columns]

print("Feature columns are " , feature_columns)
# We can use other columns later

col_to_use = ['question_title', 'question_body', 'answer', 'category']

train[col_to_use].head()
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['category'].values))

train['category'] = lbl.transform(list(train['category'].values))

lbl.fit(list(test['category'].values))

test['category'] = lbl.transform(list(test['category'].values))
# Adjusting the load_embeddings function, to now handle the pickled dict.



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words
import operator 



def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x



def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
# Lets load the embeddings 

tic = time.time()

glove_embeddings = load_embeddings(GLOVE_EMBEDDING_PATH)

print(f'loaded {len(glove_embeddings)} word vectors in {time.time()-tic}s')
# Lets check how many words we got covered 

vocab = build_vocab(list(train['question_title'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
import string

latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"

white_list = string.ascii_letters + string.digits + latin_similar + ' '

white_list += "'"
glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])

glove_symbols = ''.join([c for c in glove_chars if not c in white_list])

glove_symbols
jigsaw_chars = build_vocab(list(train["question_title"]))

jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])

jigsaw_symbols
symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])

symbols_to_delete
symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])

symbols_to_isolate
isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}



def handle_punctuation(x):

    x = x.translate(remove_dict)

    x = x.translate(isolate_dict)

    return x
train['question_title'] = train['question_title'].progress_apply(lambda x:handle_punctuation(x))

test['question_title'] = test['question_title'].progress_apply(lambda x:handle_punctuation(x))
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
def handle_contractions(x):

    x = tokenizer.tokenize(x)

    x = ' '.join(x)

    return x
train['question_title'] = train['question_title'].progress_apply(lambda x:handle_contractions(x))

test['question_title'] = test['question_title'].progress_apply(lambda x:handle_contractions(x))
def fix_quote(x):

    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

    x = ' '.join(x)

    return x
train['question_title'] = train['question_title'].progress_apply(lambda x:fix_quote(x.split()))

test['question_title'] = test['question_title'].progress_apply(lambda x:fix_quote(x.split()))
question_body_chars = build_vocab(list(train["question_body"]))

question_body_symbols = ''.join([c for c in question_body_chars if not c in white_list])

question_body_symbols
symbols_to_delete = ''.join([c for c in question_body_symbols if not c in glove_symbols])

symbols_to_delete
symbols_to_isolate = ''.join([c for c in question_body_symbols if c in glove_symbols])

symbols_to_isolate
train['question_body'] = train['question_body'].progress_apply(lambda x:handle_punctuation(x))

test['question_body'] = test['question_body'].progress_apply(lambda x:handle_punctuation(x))
# tokenize



train['question_body'] = train['question_body'].progress_apply(lambda x:handle_contractions(x))

test['question_body'] = test['question_body'].progress_apply(lambda x:handle_contractions(x))
train['question_body'] = train['question_body'].progress_apply(lambda x:fix_quote(x.split()))

test['question_body'] = test['question_body'].progress_apply(lambda x:fix_quote(x.split()))
# Check Coverage

# Lets check how many words we got covered 

vocab = build_vocab(list(train['question_body'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
answer_body_chars = build_vocab(list(train["answer"]))

answer_body_symbols = ''.join([c for c in answer_body_chars if not c in white_list])

answer_body_symbols
symbols_to_delete = ''.join([c for c in answer_body_symbols if not c in glove_symbols])

symbols_to_delete
symbols_to_isolate = ''.join([c for c in answer_body_symbols if c in glove_symbols])

symbols_to_isolate
train['answer'] = train['answer'].progress_apply(lambda x:handle_punctuation(x))

test['answer'] = test['answer'].progress_apply(lambda x:handle_punctuation(x))
# tokenize



train['answer'] = train['answer'].progress_apply(lambda x:handle_contractions(x))

test['answer'] = test['answer'].progress_apply(lambda x:handle_contractions(x))
train['answer'] = train['answer'].progress_apply(lambda x:fix_quote(x.split()))

test['answer'] = test['answer'].progress_apply(lambda x:fix_quote(x.split()))
# Check Coverage

# Lets check how many words we got covered 

vocab = build_vocab(list(train['answer'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
X = train[col_to_use]

y = train[target_cols]

test_pred = test[col_to_use]
NUM_MODELS = 1

LSTM_UNITS = 200

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 300 #220

max_features = 6000



BATCH_SIZE = 8

EPOCHS = 1
import gc

gc.collect()
tokenizer = text.Tokenizer(num_words = max_features, filters='',lower=False)

tokenizer.fit_on_texts(list(X['question_title']) + list(X['question_body'])+list(X['answer'])+ list(test_pred))
glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))



max_features = max_features or len(tokenizer.word_index) + 1

print(max_features)



embedding_matrix = np.concatenate([glove_matrix], axis=-1)

print( embedding_matrix.shape)
import gc

del glove_matrix

gc.collect()
X1 = tokenizer.texts_to_sequences(X['question_title'])

X2 = tokenizer.texts_to_sequences(X['question_body'])

X3 = tokenizer.texts_to_sequences(X['answer'])

X_cat = X['category']

test_pred1 = tokenizer.texts_to_sequences(test_pred['question_title'])

test_pred2 = tokenizer.texts_to_sequences(test_pred['question_body'])

test_pred3 = tokenizer.texts_to_sequences(test_pred['answer'])

test_cat = test_pred['category']
X_cat.shape
X1 = sequence.pad_sequences(X1, maxlen=MAX_LEN)

X2 = sequence.pad_sequences(X2, maxlen=MAX_LEN)

X3 = sequence.pad_sequences(X3, maxlen=MAX_LEN)



test_pred1 = sequence.pad_sequences(test_pred1, maxlen=MAX_LEN)

test_pred2 = sequence.pad_sequences(test_pred2, maxlen=MAX_LEN)

test_pred3 = sequence.pad_sequences(test_pred3, maxlen=MAX_LEN)
checkpoint_predictions = []

weights = []
def compute_spearmanr(trues, preds):

    rhos = []

    for col_trues, col_pred in zip(trues.T, preds.T):

        rhos.append(

            spearmanr(col_trues, col_pred))

    return(np.mean(rhos))
import tensorflow as tf
class CustomCallback(tf.keras.callbacks.Callback):

    

    def __init__(self, valid_data,batch_size=16, fold=None):

        

        self.X1_val = valid_data[0][0]

        self.X2_val = valid_data[0][1]

        self.X3_val = valid_data[0][2]

        self.X_cat_val = valid_data[0][3]

        self.valid_outputs = valid_data[1]

        

        self.batch_size = batch_size

        self.fold = fold

        

    def on_train_begin(self, logs={}):

        self.valid_predictions = []

        self.test_predictions = []

        

    def on_epoch_end(self, epoch, logs={}):

        self.valid_predictions.append(

            self.model.predict(([self.X1_val,self.X2_val,self.X3_val,self.X_cat_val],self.valid_outputs), batch_size=self.batch_size))

        

        rho_val = compute_spearmanr(

            self.valid_outputs, np.average(self.valid_predictions, axis=0))

        

        print("\nvalidation rho: %.4f" % rho_val)

        

        if self.fold is not None:

            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate,Flatten,Lambda

from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,PReLU,LSTM

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from keras.models import Sequential

from keras.preprocessing import text, sequence

from keras import regularizers

import keras.backend as K

from sklearn.model_selection import train_test_split

from keras.engine.topology import Layer

import tensorflow_hub as hub

from keras.layers.normalization import BatchNormalization

from keras.layers import Concatenate
X1_train , X1_val,X2_train, X2_val,X3_train, X3_val,X_cat_train,X_cat_val ,y_train  , y_val = train_test_split(X1 , X2,X3,X_cat,

                                                     y , 

                                                     train_size = 0.8,

                                                     random_state = 100)
print(X1_train.shape)

print(X2_train.shape)

print(X3_train.shape)
print(X1_val.shape)

print(X2_val.shape)

print(X3_val.shape)
from keras.callbacks import EarlyStopping 

es = EarlyStopping(monitor='val_loss', mode ='min' ,verbose =1)
def build_model(embedding_matrix, num_aux_targets):

    title = Input(shape=(MAX_LEN,))

    question_body = Input(shape=(MAX_LEN,))

    answer = Input(shape=(MAX_LEN,))

    category = Input(shape=(1,))

    

    title_embb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False )(title)

    question_body_embb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(question_body)

    answer_embb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(answer)

    concat = Concatenate(axis=1)

    embb_final = concat([title_embb,question_body_embb,answer_embb])

    

    x1 = SpatialDropout1D(0.3)(embb_final)

    x1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x1)

    x1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x1)

    hidden1 = concatenate([

        GlobalMaxPooling1D()(x1), 

        GlobalAveragePooling1D()(x1),#layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input 

        #of variable length in the simplest way possible.

    ])

    hidden1 = add([hidden1, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden1)])

    hidden1 = Dense(30, activation='sigmoid')(hidden1)

    category1 = Dense(30, activation='sigmoid')(category)

    

    final = add([hidden1,category1])

    

    result = Dense(30, activation='sigmoid')(final)

    model = Model(inputs=[title,question_body,answer,category], outputs= result)

    model._name = 'mymodel'

    model.compile(loss='binary_crossentropy',metrics = ['accuracy'], optimizer='adam')

    model.summary()

    return model
for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix,1)

    for global_epoch in range(EPOCHS):      

        model.fit(

            [X1_train,X2_train,X3_train,X_cat_train],

            y_train,

            validation_data = ([X1_val,X2_val,X3_val,X_cat_val], y_val),

            batch_size=BATCH_SIZE,

            epochs=4,

            verbose=2,

            callbacks=[

                LearningRateScheduler(lambda epoch: 0.4 * (0.1 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict([test_pred1,test_pred2,test_pred3,test_cat]).flatten())

        weights.append(2 ** global_epoch)
predictions = model.predict([test_pred1,test_pred2,test_pred3,test_cat])
sample_submission.iloc[:, 1:] = predictions

sample_submission.to_csv('submission.csv', index=False)