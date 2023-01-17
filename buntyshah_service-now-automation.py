!pip show tensorflow
import numpy as np

import pandas as pd

import tensorflow

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.callbacks import LearningRateScheduler

import time

import pickle

import re

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
train = pd.read_csv('../input/hackathonservicenowautomation/all_tickets.csv')

train.head()
train['impact'].value_counts()
train_urgency = train[['title','body','urgency']]
train_urgency.head()
#combine 2 columns , title and body



train_urgency['text'] = train_urgency['title'] + " " + train_urgency['body']
train_urgency = train_urgency.drop(['title','body'],axis=1)
train_urgency = train_urgency.fillna('No Data')
#Handle URL



train_urgency['text'] = train_urgency['text'].apply(lambda x: re.sub(r'http\S+', '', x))
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



vocab = build_vocab(list(train_urgency['text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
import gc

gc.collect()
import string

latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"

white_list = string.ascii_letters + string.digits + latin_similar + ' '

white_list += "'"
glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])

glove_symbols = ''.join([c for c in glove_chars if not c in white_list])

glove_symbols
jigsaw_chars = build_vocab(list(train_urgency['text']))

jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])

jigsaw_symbols
# Basically we can delete all symbols we have no embeddings for:



symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])

symbols_to_delete
# The symbols we want to keep we need to isolate from our words. So lets setup a list of those to isolate.



symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])

symbols_to_isolate
# Note : Next comes the next trick. Instead of using an inefficient loop of replace we use translate. 

# I find the syntax a bit weird, but the improvement in speed is worth the worse readablity. 



isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}





def handle_punctuation(x):

    x = x.translate(remove_dict)

    x = x.translate(isolate_dict)

    return x
#So lets apply that function to our text and reasses the coverage



train_urgency['text'] = train_urgency['text'].apply(lambda x:handle_punctuation(x))
# remove whitespaces

train_urgency['text'] = train_urgency['text'].apply(lambda x:' '.join(x.split()))
vocab = build_vocab(list(train_urgency['text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:10]
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
def handle_contractions(x):

    x = tokenizer.tokenize(x)

    x = ' '.join(x)

    return x
train_urgency['text'] = train_urgency['text'].apply(lambda x:handle_contractions(x))
vocab = build_vocab(list(train_urgency['text'].apply(lambda x:x.split())),verbose=False)

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
def fix_quote(x):

    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

    x = ' '.join(x)

    return x
train_urgency['text'] = train_urgency['text'].apply(lambda x:fix_quote(x.split()))
tic = time.time()

crawl_embeddings = load_embeddings(CRAWL_EMBEDDING_PATH)

print(f'loaded {len(glove_embeddings)} word vectors in {time.time()-tic}s')
vocab = build_vocab(list(train_urgency['text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,crawl_embeddings)

oov[:20]
X = train_urgency['text']

y = train_urgency['urgency']
NUM_MODELS = 2

LSTM_UNITS = 200

DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS

MAX_LEN = 600 #220

max_features = 200000



BATCH_SIZE = 50  #changed from 150

EPOCHS = 4
# Its really important that you intitialize the keras tokenizer correctly. Per default it does lower case and removes a lot of symbols. We want neither of that!



tokenizer = text.Tokenizer(num_words = max_features, filters='',lower=False)
tokenizer.fit_on_texts(list(X))
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))



max_features = max_features or len(tokenizer.word_index) + 1

max_features



embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape



import gc

del crawl_matrix

del glove_matrix

gc.collect()
X = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen=MAX_LEN)
checkpoint_predictions = []

weights = []
# Check F1 score



from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate,Flatten,Lambda

from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,PReLU,LSTM

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split

import tensorflow_hub as hub
X_train , X_val, y_train  , y_val = train_test_split(X , 

                                                     y , 

                                                     stratify = y.values , 

                                                     train_size = 0.8,

                                                     random_state = 100)
unique = train_urgency['urgency'].nunique()
unique
from tensorflow.keras.callbacks import EarlyStopping 

es = EarlyStopping(monitor='val_loss', mode ='min' ,verbose =1)
train_urgency['urgency'].value_counts()
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words) #Finds word embeddings for each word

    x = SpatialDropout1D(0.3)(x) #This version performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements

    x = LSTM(LSTM_UNITS, return_sequences=True)(x)

    x = LSTM(LSTM_UNITS, return_sequences=True)(x)

    x = LSTM(LSTM_UNITS, return_sequences=True)(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x), 

        GlobalAveragePooling1D()(x),#layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input 

        #of variable length in the simplest way possible.

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)]) #This fixed-length output vector is piped through a fully-connected (Dense) layer with x hidden units.

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(unique, activation='softmax')(hidden)

    model = Model(inputs=words, outputs= result)

    model.compile(loss='sparse_categorical_crossentropy',metrics = ['accuracy'], optimizer='adam')

    model.summary()

    return model
type(y_val)
model = build_model(embedding_matrix,1)



model.fit(

    X_train,

    np.asarray(y_train),

    validation_data = (X_val, np.asarray(y_val)),

    batch_size=BATCH_SIZE,

    epochs=100,

    verbose=2,

    callbacks=[

        LearningRateScheduler(lambda epoch: 1e-3 * (1 ** 2)),

        es

    ]

)
y_pred = model.predict(X_val)
y_pred =  np.argmax(y_pred,axis=1)
from sklearn.metrics import classification_report

print(classification_report(y_val,y_pred))
from PIL import Image

from IPython.display import display, HTML, clear_output

from ipywidgets import widgets, Layout
def init_widgets():

    text_subject = widgets.Text(

    description="Subject", layout=Layout(minwidth="70%")

  )



    text_body = widgets.Text(

    description="Body", layout=Layout(minwidth="70%")

  )

    submit_button = widgets.Button(description="Submit")



    display(text_subject)

    display(text_body)

    display(submit_button)



    prediction = submit_button.on_click(lambda b: on_button_click(

      b,text_subject,text_body

  ))

  #display(prediction)

    return prediction,text_subject,text_body
def on_button_click(b,text_subject,text_body):

    clear_output()

    subject = text_subject.value

    body = text_body.value



    text = subject + " " + body



    tokenizer.fit_on_texts(list(text))

  

    X_text = pd.DataFrame()

    X_text['text'] = text

    X_text = tokenizer.texts_to_sequences(X_text)

    X_text = sequence.pad_sequences(X_text, maxlen=MAX_LEN)



    #display(X_text)

    pred = model.predict(X_text)

    pred =  np.argmax(pred,axis=1)

    display(pred[0])

    return pred[0]
prediction,text_subject,text_body = init_widgets()
test = pd.read_csv('../input/hackathonservicenowautomation/test_tickets.csv')
test['text'] = test['title'] + " " + test['body']

test = test.drop(['title','body'],axis=1)

test = test.fillna('No Data')

test['text'] = test['text'].apply(lambda x: re.sub(r'http\S+', '', x))
test = tokenizer.texts_to_sequences(test['text'])

test = sequence.pad_sequences(test, maxlen=MAX_LEN)
prediction  = loaded_model.predict(test)
prediction =  np.argmax(prediction,axis=1)