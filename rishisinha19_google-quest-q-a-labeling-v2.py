# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras.models import *

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

import warnings 

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'
sample_submission_df = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

test_df = pd.read_csv("../input/google-quest-challenge/test.csv")

train_df = pd.read_csv("../input/google-quest-challenge/train.csv")
sample_submission_df.shape
train_df.shape
# Exploring Data

samp_id = 9

print('Question Title: %s \n' % train_df['question_title'].values[samp_id])

print('Question Body: %s \n' % train_df['question_body'].values[samp_id])

print('Answer: %s' % train_df['answer'].values[samp_id])
train_users = set(train_df['question_user_page'].unique())

test_users = set(test_df['question_user_page'].unique())



print('Unique users in train set: %s' % len(train_users))

print('Unique users in test set: %s' % len(test_users))

print('Users in both sets: %s' % len(train_users & test_users))

print('What users are in both sets? %s' % list(train_users & test_users))
train_df.head()
(train_df.head()).T
#!pip install transformers
train_df['all_text'] = train_df['question_title'] + " [SEP] " + train_df['question_body'] + " [SEP] " + train_df['question_user_name'] + " [SEP] " + train_df['question_user_page'] + " [SEP] " + train_df['question_user_page'] + " [SEP] " + train_df['answer'] + " [SEP] " + train_df['answer_user_name'] + " [SEP] " + train_df['answer_user_page'] + " [SEP] " + train_df['category'] + " [SEP] " + train_df['host'] 



test_df['all_text'] = test_df['question_title'] + " [SEP] " + test_df['question_body'] + " [SEP] " + test_df['question_user_name'] + " [SEP] " + test_df['question_user_page'] + " [SEP] " + test_df['question_user_page'] + " [SEP] " + test_df['answer'] + " [SEP] " + test_df['answer_user_name'] + " [SEP] " + test_df['answer_user_page'] + " [SEP] " + test_df['category'] + " [SEP] " + test_df['host'] 
# You can use this pre-processing steps at later stage



import re

def pre_process(text):

    new_text =re.sub('[0-9]', '', text)

    new_text = re.sub(r"\u200b","",new_text)

    new_text = re.sub(r"\.+",".",new_text)

    new_text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '',new_text, flags=re.MULTILINE)

    new_text = re.sub("'", "", new_text)

    new_text = re.sub(r'↑', '', new_text)

    new_text = re.sub("\t", "", new_text)

    new_text = re.sub("\xa0", "", new_text)

    new_text = re.sub("\(|\)|\[|\]", "", new_text)

    new_text = re.sub("\n", "", new_text)

    new_text = re.sub("\.", "", new_text)

    new_text = re.sub("\,", " ", new_text)

    new_text = re.sub("[/%]", " ", new_text)

    new_text = re.sub('[/%:;]', '', new_text)

    new_text = re.sub(' +', ' ', new_text)

    return new_text
# remove URL's from train and test

train_df['all_text'] = train_df['all_text'].apply(lambda x: re.sub(r'http\S+', '', x))

test_df['all_text'] = test_df['all_text'].apply(lambda x: re.sub(r'http\S+', '', x))
# remove numbers

train_df['all_text'] = train_df['all_text'].str.replace("[0-9]", " ")

test_df['all_text'] = test_df['all_text'].str.replace("[0-9]", " ")
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
#Below Functions are to check the embedding’s coverage and building vocabulary 

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

vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
# Lets replace few words which is not covered in our embedding’s.



replaceWords1 = { "won't":"will not","$&@*#":"in most profane vulgar shitty terms","#$&@*#":"shitty",

 "can't":"cannot","aren't": 'are not',

 "Aren't": 'Are not',

 "AREN'T": 'ARE NOT',

 "C'est": "C'est",

 "C'mon": "C'mon",

 "c'mon": "c'mon",

 "can't": 'cannot',

 "Can't": 'Cannot',

 "CAN'T": 'CANNOT',

 "con't": 'continued',

 "cont'd": 'continued',

 "could've": 'could have',

 "couldn't": 'could not',

 "Couldn't": 'Could not',

 "didn't": 'did not',

 "Didn't": 'Did not',

 "DIDN'T": 'DID NOT',

 "don't": 'do not',

 "Don't": 'Do not',

 "DON'T": 'DO NOT',

 "doesn't": 'does not',

 "Doesn't": 'Does not',

 "else's": 'else',

 "gov's": 'government',

 "Gov's": 'government',

 "gov't": 'government',

 "Gov't": 'government',

 "govt's": 'government',

 "gov'ts": 'governments',

 "hadn't": 'had not',

 "hasn't": 'has not',

 "Hasn't": 'Has not',

 "haven't": 'have not',

 "Haven't": 'Have not',

 "he's": 'he is',

 "He's": 'He is',

 "he'll": 'he will',

 "He'll": 'He will',

 "he'd": 'he would',

 "He'd": 'He would',

 "Here's": 'Here is',

 "here's": 'here is',

 "I'm": 'I am',

 "i'm": 'i am',

 "I'M": 'I am',

 "I've": 'I have',

 "i've": 'i have',

 "I'll": 'I will',

 "i'll": 'i will',

 "I'd": 'I would',

 "i'd": 'i would',

 "ain't": 'is not',

 "isn't": 'is not',

 "Isn't": 'Is not',

 "ISN'T": 'IS NOT',

 "it's": 'it is',

 "It's": 'It is',

 "IT'S": 'IT IS',

 "I's": 'It is',

 "i's": 'it is',

 "it'll": 'it will',

 "It'll": 'It will',

 "it'd": 'it would',

 "It'd": 'It would',

 "Let's": "Let's",

 "let's": 'let us',

 "ma'am": 'madam',

 "Ma'am": "Madam",

 "she's": 'she is',

 "She's": 'She is',

 "she'll": 'she will',

 "She'll": 'She will',

 "she'd": 'she would',

 "She'd": 'She would',

 "shouldn't": 'should not',

 "that's": 'that is',

 "That's": 'That is',

 "THAT'S": 'THAT IS',

 "THAT's": 'THAT IS',

 "that'll": 'that will',

 "That'll": 'That will',

 "there's": 'there is',

 "There's": 'There is',

 "there'll": 'there will',

 "There'll": 'There will',

 "there'd": 'there would',

 "they're": 'they are',

 "They're": 'They are',

 "they've": 'they have',

 "They've": 'They Have',

 "they'll": 'they will',

 "They'll": 'They will',

 "they'd": 'they would',

 "They'd": 'They would',

 "wasn't": 'was not',

 "we're": 'we are',

 "We're": 'We are',

 "we've": 'we have',

 "We've": 'We have',

 "we'll": 'we will',

 "We'll": 'We will',

 "we'd": 'we would',

 "We'd": 'We would',

 "What'll": 'What will',

 "weren't": 'were not',

 "Weren't": 'Were not',

 "what's": 'what is',

 "What's": 'What is',

 "When's": 'When is',

 "Where's": 'Where is',

 "where's": 'where is',

 "Where'd": 'Where would',

 "who're": 'who are',

 "who've": 'who have',

 "who's": 'who is',

 "Who's": 'Who is',

 "who'll": 'who will',

 "who'd": 'Who would',

 "Who'd": 'Who would',

 "won't": 'will not',

 "Won't": 'will not',

 "WON'T": 'WILL NOT',

 "would've": 'would have',

 "wouldn't": 'would not',

 "Wouldn't": 'Would not',

 "would't": 'would not',

 "Would't": 'Would not',

 "y'all": 'you all',

 "Y'all": 'You all',

 "you're": 'you are',

 "You're": 'You are',

 "YOU'RE": 'YOU ARE',

 "you've": 'you have',

 "You've": 'You have',

 "y'know": 'you know',

 "Y'know": 'You know',

 "ya'll": 'you will',

 "you'll": 'you will',

 "You'll": 'You will',

 "you'd": 'you would',

 "You'd": 'You would',

 "Y'got": 'You got',

 'cause': 'because',

 "had'nt": 'had not',

 "Had'nt": 'Had not',

 "how'd": 'how did',

 "how'd'y": 'how do you',

 "how'll": 'how will',

 "how's": 'how is',

 "I'd've": 'I would have',

 "I'll've": 'I will have',

 "i'd've": 'i would have',

 "i'll've": 'i will have',

 "it'd've": 'it would have',

 "it'll've": 'it will have',

 "mayn't": 'may not',

 "might've": 'might have',

 "mightn't": 'might not',

 "mightn't've": 'might not have',

 "must've": 'must have',

 "mustn't": 'must not',

 "mustn't've": 'must not have',

 "needn't": 'need not',

 "needn't've": 'need not have',

 "o'clock": 'of the clock',

 "oughtn't": 'ought not',

 "oughtn't've": 'ought not have',

 "shan't": 'shall not',

 "sha'n't": 'shall not',

 "shan't've": 'shall not have',

 "she'd've": 'she would have',

 "she'll've": 'she will have',

 "should've": 'should have',

 "shouldn't've": 'should not have',

 "so've": 'so have',

 "so's": 'so as',

 "this's": 'this is',

 "that'd": 'that would',

 "that'd've": 'that would have',

 "there'd've": 'there would have',

 "they'd've": 'they would have',

 "they'll've": 'they will have',

 "to've": 'to have',

 "we'd've": 'we would have',

 "we'll've": 'we will have',

 "what'll": 'what will',

 "what'll've": 'what will have',

 "what're": 'what are',

 "what've": 'what have',

 "when's": 'when is',

 "when've": 'when have',

 "where'd": 'where did',

 "where've": 'where have',

 "who'll've": 'who will have',

 "why's": 'why is',

 "why've": 'why have',

 "will've": 'will have',

 "won't've": 'will not have',

 "wouldn't've": 'would not have',

 "y'all'd": 'you all would',

 "y'all'd've": 'you all would have',

 "y'all're": 'you all are',

 "y'all've": 'you all have',

 "you'd've": 'you would have',

 "you'll've": 'you will have',

'bebecause':'be because',

'I’m':'I am',

              'it’s':'it is',

                 'I’ve':'I have',

                 'don’t':'do not',

                'However':'but',

                 'It’s':'It is',

                 'didn’t':'did not',

                 'can’t':'can not',

                 'that’s':'that is',

'doesn’t':'does not',

'I’d':'I had',

'isn’t':'is not',

'wasn’t':'was not'

                

                }



def wordreplace(tweet,replaceWords):

    for key in replaceWords:

        tweet = tweet.replace(key,replaceWords[key])

    return tweet



for index, row in train_df['all_text'].iteritems():

    train_df['all_text'][index] = wordreplace(row,replaceWords1)

    

for index, row in test_df['all_text'].iteritems():

    test_df['all_text'][index] = wordreplace(row,replaceWords1)
# Now lets check if we have improved on our coverage 

vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
import string

latin_similar = "’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"

white_list = string.ascii_letters + string.digits + latin_similar + ' '

white_list += "'"
glove_chars = ''.join([c for c in tqdm(glove_embeddings) if len(c) == 1])

glove_symbols = ''.join([c for c in glove_chars if not c in white_list])

glove_symbols
# Chars available in the embedding’s

jigsaw_chars = build_vocab(list(train_df["all_text"]))

jigsaw_symbols = ''.join([c for c in jigsaw_chars if not c in white_list])

jigsaw_symbols
# Basically we can delete all symbols we have no embeddings for:

symbols_to_delete = ''.join([c for c in jigsaw_symbols if not c in glove_symbols])

symbols_to_delete
# The symbols we want to keep we need to isolate from our words. So lets setup a list of those to isolate.

symbols_to_isolate = ''.join([c for c in jigsaw_symbols if c in glove_symbols])

symbols_to_isolate
# Note : Next comes the next trick. Instead of using an inefficient loop of replace we use translate. 





isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}





def handle_punctuation(x):

    x = x.translate(remove_dict)

    x = x.translate(isolate_dict)

    return x
#So lets apply that function to our text and reasses the coverage



train_df['all_text'] = train_df['all_text'].progress_apply(lambda x:handle_punctuation(x))

test_df['all_text'] = test_df['all_text'].progress_apply(lambda x:handle_punctuation(x))
vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
# Lets apply pre-processing before we tokenize the words



for index, row in train_df['all_text'].iteritems():

    train_df['all_text'][index] = pre_process(row)

for index, row in test_df['all_text'].iteritems():

    test_df['all_text'][index] = pre_process(row)
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
def handle_contractions(x):

    x = tokenizer.tokenize(x)

    x = ' '.join(x)

    return x

train_df['all_text'] = train_df['all_text'].progress_apply(lambda x:handle_contractions(x))

test_df['all_text'] = test_df['all_text'].progress_apply(lambda x:handle_contractions(x))
# Lets check after we have tokenize



vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())),verbose=False)

oov = check_coverage(vocab,glove_embeddings)

oov[:20]
def fix_quote(x):

    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

    x = ' '.join(x)

    return x
train_df['all_text'] = train_df['all_text'].progress_apply(lambda x:fix_quote(x.split()))

test_df['all_text'] = test_df['all_text'].progress_apply(lambda x:fix_quote(x.split()))
vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())),verbose=False)

oov = check_coverage(vocab,glove_embeddings)

oov[:10]
# Lets also check test data has equal coverage

vocab = build_vocab(list(test_df['all_text'].apply(lambda x:x.split())),verbose=False)

oov = check_coverage(vocab,glove_embeddings)

oov[:10]
tic = time.time()

crawl_embeddings = load_embeddings(CRAWL_EMBEDDING_PATH)

print(f'loaded {len(glove_embeddings)} word vectors in {time.time()-tic}s')
vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,crawl_embeddings)

oov[:20]
punctuation = '_`'



train_df['all_text'] = train_df['all_text'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

test_df['all_text'] = test_df['all_text'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
#Lets check the embeddings now



vocab = build_vocab(list(train_df['all_text'].apply(lambda x:x.split())))

oov = check_coverage(vocab,crawl_embeddings)

oov[:10]
X = train_df['all_text']

y_columns = ['question_asker_intent_understanding',

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

#y = train_df['question_asker_intent_understanding']

y = train_df[y_columns]

#y = train_df['question_fact_seeking']

test_pred = test_df['all_text']

train_df[y_columns].head(n=10)
MAX_LEN = 100

max_features = 500000


# Its really important that you intitialize the keras tokenizer correctly. Per default it does lower case and removes a lot of symbols. We want neither of that!



tokenizer = text.Tokenizer(num_words = max_features, filters='',lower=False)
tokenizer.fit_on_texts(list(X) + list(test_pred))
import numpy as np

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

test_pred = tokenizer.texts_to_sequences(test_pred)
X = sequence.pad_sequences(X, maxlen=MAX_LEN)

test_pred = sequence.pad_sequences(test_pred, maxlen=MAX_LEN)
checkpoint_predictions = []

weights = []
from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate,Flatten,Lambda

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D,PReLU,LSTM

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

from keras.models import Sequential

from keras.preprocessing import text, sequence

from keras import regularizers

import keras

import tensorflow as tf

import keras.backend as K

from sklearn.model_selection import train_test_split

from keras.engine.topology import Layer

import tensorflow_hub as hub

from keras.layers.normalization import BatchNormalization
#### testing delete
X_train , X_val, y_train  , y_val = train_test_split(X , 

                                                     y.values , 

                                                     #stratify = y.values, 

                                                     train_size = 0.8,

                                                     random_state = 100)
y_train.shape
embedding_matrix.shape
len(X_train[1])
from scipy.stats import spearmanr, rankdata

from keras.callbacks import Callback

# Compatible with tensorflow backend

class SpearmanRhoCallback(Callback):

    def __init__(self, training_data, validation_data, patience, model_name):

        self.x = training_data[0]

        self.y = training_data[1]

        self.X_val = validation_data[0]

        self.y_val = validation_data[1]

        

        self.patience = patience

        self.value = -1

        self.bad_epochs = 0

        self.model_name = model_name



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.X_val)

        ind = 0

        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])

        if rho_val >= self.value:

            self.value = rho_val

        else:

            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:

            print("Epoch %05d: early stopping Threshold" % epoch)

            self.model.stop_training = True

            #self.model.save_weights(self.model_name)

        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')

        return rho_val



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
maxlen = 100

vocab_size = len(tokenizer.word_index) + 1

embedding_dim = 100

num_class = len(np.unique(y))

def build_model(embedding_matrix, num_aux_targets):

        model = tf.keras.models.Sequential([

        tf.keras.layers.Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv1D(150, 5, activation='relu'),

        tf.keras.layers.LSTM(128, return_sequences=True),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(units=120,activation='relu'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(30, activation='sigmoid')])

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),

             loss = 'binary_crossentropy',

             metrics = ['accuracy'])

        print(model.summary())

        return model                         
from keras.callbacks import EarlyStopping 

es = EarlyStopping(monitor='val_loss', mode ='min' ,verbose =1,patience=0.1)

NUM_MODELS =2

for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix,1)

    for global_epoch in range(200):

         model.fit(X_train,

                   y_train,

                   epochs=200,

                   verbose=1,

                   batch_size = 50,

                   validation_data=(X_val, y_val),

                   class_weight ="temporal",#class_weights,

                   callbacks=[SpearmanRhoCallback(training_data=(X_train, y_train), validation_data=(X_val, y_val),

                                       patience=5, model_name=f'pre_trained.hdf5'),

                              LearningRateScheduler(lambda epoch: 1e-3 * (0.4 ** global_epoch)),

                       es

                   ]

                  )

                   

    checkpoint_predictions.append(model.predict(test_pred))

    weights.append(3 ** global_epoch)
score = model.evaluate(X_val, y_val, verbose=1)

print("Test Score:", score[0])

print("Test Accuracy:", score[1])
predictions = np.average(checkpoint_predictions,  axis=0) 
predictions
sample_submission_df.iloc[:, 1:] = predictions
sample_submission_df.head(n=6)
sample_submission_df.to_csv("submission.csv", index= False)