import os, warnings, pickle, gc, re, string



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf



from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers



from tensorflow.keras.layers import Layer, Dense, Input, Activation, Embedding, SpatialDropout1D, Bidirectional, LSTM, GRU, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout

from tensorflow.keras.layers import concatenate, add



from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler, EarlyStopping

from tensorflow.keras import backend as K



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



from keras.callbacks import ModelCheckpoint



from tqdm.notebook import tqdm

tqdm.pandas()



from sklearn.metrics import roc_auc_score



warnings.simplefilter('ignore')
# HYPERPARAMETERS



MAX_LEN = 1024

MAX_FEATURES = 100000

EMBED_SIZE = 600



BATCH_SIZE = 32

N_EPOCHS = 3



LEARNING_RATE = 8e-4



# We will concatenate Crawl and GloVe embeddings



CRAWL_EMB_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'

GLOVE_EMB_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'
def display_training_curves(training, validation, title, subplot):

    '''

    Quickly display training curves

    '''

    if subplot % 10 == 1:

        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')

        plt.tight_layout()

    

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model' + title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid'])
def get_coeffs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(embed_dir):

    with open(embed_dir, 'rb') as  infile:

        embeddings = pickle.load(infile)

        return embeddings
# def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

#     embedding_matrix = np.zeros((max_features, 300))

#     for word, i in tqdm(word_index.items(), len=(word_index.items())):

#         if lower:

#             word = word.lower()

#         if i >= max_features: continue

#         try:

#             embedding_vector = embeddings_index[word]

#         except:

#             embedding_vector = embeddings_index["unknown"]

#         if embedding_vector is not None:

#             # words not found in embedding index will be all-zeros.

#             embedding_matrix[i] = embedding_vector

#     return embedding_matrix
# def build_matrix(word_index, embeddings_index):

#     embedding_matrix = np.zeros((len(word_index) + 1,300))

#     for word, i in word_index.items():

#         try:

#             embedding_matrix[i] = embeddings_index[word]

#         except:

#             embedding_matrix[i] = embeddings_index["unknown"]

#     return embedding_matrix
class Attention(Layer):

    """

    Custom Keras attention layer

    Reference: https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043

    """

    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, 

                 W_constraint=None, b_constraint=None, bias=True, **kwargs):



        self.supports_masking = True



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = None

        super(Attention, self).__init__(**kwargs)



        self.param_W = {

            'initializer': initializers.get('glorot_uniform'),

            'name': '{}_W'.format(self.name),

            'regularizer': regularizers.get(W_regularizer),

            'constraint': constraints.get(W_constraint)

        }

        self.W = None



        self.param_b = {

            'initializer': 'zero',

            'name': '{}_b'.format(self.name),

            'regularizer': regularizers.get(b_regularizer),

            'constraint': constraints.get(b_constraint)

        }

        self.b = None



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.features_dim = input_shape[-1]

        self.W = self.add_weight(shape=(input_shape[-1],), 

                                 **self.param_W)



        if self.bias:

            self.b = self.add_weight(shape=(input_shape[1],), 

                                     **self.param_b)



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        step_dim = self.step_dim

        features_dim = self.features_dim



        eij = K.reshape(

            K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),

            (-1, step_dim))



        if self.bias:

            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0], self.features_dim
# We create a balanced



print('Loading train sets...')

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")



train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

])



del train1, train2



print('Loading validation sets...')

valid = pd.read_csv('/kaggle/input/val-en-df/validation_en.csv')



print('Loading test sets...')

test = pd.read_csv('/kaggle/input/test-en-df/test_en.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",

                 "didn't": "did not", "doesn't": "does not", "don't": "do not",

                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                 "he'd": "he would", "he'll": "he will", "he's": "he is",

                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",

                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",

                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",

                 "she'd": "she would", "she'll": "she will", "she's": "she is",

                 "shouldn't": "should not", "that's": "that is", "there's": "there is",

                 "they'd": "they would", "they'll": "they will", "they're": "they are",

                 "they've": "they have", "we'd": "we would", "we're": "we are",

                 "weren't": "were not", "we've": "we have", "what'll": "what will",

                 "what're": "what are", "what's": "what is", "what've": "what have",

                 "where's": "where is", "who'd": "who would", "who'll": "who will",

                 "who're": "who are", "who's": "who is", "who've": "who have",

                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",

                 "you'll": "you will", "you're": "you are", "you've": "you have",

                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}





def _get_misspell(misspell_dict):

    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    return misspell_dict, misspell_re





def replace_typical_misspell(text):

    misspellings, misspellings_re = _get_misspell(misspell_dict)



    def replace(match):

        return misspellings[match.group(0)]



    return misspellings_re.sub(replace, text)

    



puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',

          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',

          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',

          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',

          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',

          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',

          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']





def clean_text(x):

    x = str(x)

    for punct in puncts + list(string.punctuation):

        if punct in x:

            x = x.replace(punct, f' {punct} ')

    return x





def clean_numbers(x):

    return re.sub(r'\d+', ' ', x)
def preprocess(train, valid, test, tfms):

    for tfm in tfms:

        print(tfm.__name__)

        train['comment_text'] = train['comment_text'].progress_apply(tfm)

        valid['comment_text_en'] = valid['comment_text_en'].progress_apply(tfm)

        test['content'] = test['content'].progress_apply(tfm)

    

    return train, valid, test
tfms = [replace_typical_misspell, clean_text, clean_numbers]

train, valid, test = preprocess(train, valid, test, tfms)
tokenizer = Tokenizer(num_words=MAX_FEATURES, filters='', lower=False)



print('Fitting tokenizer...')

tokenizer.fit_on_texts(list(train['comment_text']) + list(valid['comment_text_en']) + list(test['content_en']))

word_index = tokenizer.word_index



print('Building training set...')

X_train = tokenizer.texts_to_sequences(list(train['comment_text']))

y_train = train['toxic'].values



print('Building validation set...')

X_valid = tokenizer.texts_to_sequences(list(valid['comment_text_en']))

y_valid = valid['toxic'].values



print('Building test set ...')

X_test = tokenizer.texts_to_sequences(list(test['content_en']))



print('Padding sequences...')

X_train = pad_sequences(X_train, maxlen=MAX_LEN)

X_valid = pad_sequences(X_valid, maxlen=MAX_LEN)

X_test = pad_sequences(X_test, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values



del tokenizer
embedding_matrix = np.load('/kaggle/input/jigsaw-ml-toxic-comments-laser-embeddings/jigsaw-ml_laser.npy')



# import tensorflow as tf

# from transformers import BertTokenizer, TFBertModel



# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model = TFBertModel.from_pretrained('bert-base-uncased')

# input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1

# outputs = model(input_ids)

# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(X_test)

    .batch(BATCH_SIZE)

)
def build_model(embedding_matrix, verbose=True):

    '''

    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/

    '''

    sequence_input = Input(shape=(MAX_LEN,), dtype=tf.int32)

    

    embedding_layer = Embedding(*embedding_matrix.shape,

                                weights=[embedding_matrix],

                                trainable=False)

    

    x = embedding_layer(sequence_input)

    x = SpatialDropout1D(0.3)(x)

    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    

    att = Attention(MAX_LEN)(x)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)

    hidden = concatenate([att, avg_pool1, max_pool1])

    

    hidden = Dense(512, activation='relu')(hidden)

    hidden = Dense(128, activation='relu')(hidden)



    out = Dense(1, activation='sigmoid')(hidden)

    

    model = Model(sequence_input, out)

    

    return model
model = build_model(embedding_matrix)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

model.summary()
file_weights = 'best_model.h5'

#cb1 = ModelCheckpoint(file_weights, save_best_only=True)



cb2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)



cb3 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, cooldown=0, min_lr=0.0001)



cb4 = LearningRateScheduler(lambda epoch: LEARNING_RATE * (0.6 ** epoch))
n_steps = X_train.shape[0] // BATCH_SIZE



train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    callbacks=[cb4],

    epochs=N_EPOCHS

)
display_training_curves(

    train_history.history['loss'],

    train_history.history['val_loss'],

    'loss',

    211)



display_training_curves(

    train_history.history['auc'],

    train_history.history['val_auc'],

    'AUC',

    212)
n_steps = X_valid.shape[0] // BATCH_SIZE



train_history = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    callbacks=[cb4],

    epochs=N_EPOCHS

)
preds = model.predict(test_dataset, verbose=1)

sub['toxic'] = preds
sub.to_csv('submission.csv', index=False)