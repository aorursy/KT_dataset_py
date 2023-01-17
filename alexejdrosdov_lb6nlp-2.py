import pandas as pd

from tqdm import tqdm

import numpy as np

tqdm.pandas()



import fastText as fasttext

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import f1_score, roc_auc_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import concatenate

from keras.callbacks import *

from keras.initializers import *

from keras.optimizers import *

from keras.callbacks import *

from keras.layers import *

from keras.models import *

from nltk.tag import pos_tag

from xgboost import XGBClassifier

from gensim.models import KeyedVectors
## some config values 

embed_size = 300 # how big is each word vector

max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use

TOP_N_FIRST_WORDS = 40 #originally 40

NUM_CUSTOM_FEATURES = 15 + TOP_N_FIRST_WORDS

NUM_TOTAL_FEATURES = 19 #after dropping nonimportant features
#Loads embeddings

glove_EMBEDDING_FILE = open('/kaggle/input/popular-embedding/glove.840B.300d/glove.840B.300d.txt')

para_embedding_file = open('/kaggle/input/popular-embedding/paragram_300_sl999/paragram_300_sl999.txt', encoding="utf8", errors='ignore')



def load_glove(word_index):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in glove_EMBEDDING_FILE)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #unique_words = [None] * max_features

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        #unique_words[i] = word

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 

    

def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/popular_embedding/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in KeyedVectors.load_word2vec_format(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #unique_words = [None] * max_features

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        #unique_words[i] = word

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix



def load_para(word_index):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in para_embedding_file if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    #unique_words = [None] * max_features

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        #unique_words[i] = word

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    

    return embedding_matrix
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Text Cleaning



import re



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    x = re.sub('[0-9]k', '# thousand', x)

    return x
# Text Cleaning

def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color',

                'grey' : 'gray',

                'recognised' : 'recognized',

                'recognise' : 'recognize',

                'defence' : 'defense',

                'programmes' : 'programme',

                'centre' : 'center',

                'didnt' : 'did not',

                'doesnt' : 'does not',

                'isnt' : 'is not',

                'Isnt' : 'is not',

                'hasnt' : 'has not',

                'wasnt' : 'was not',

                'Doesnt' : 'does not',

                'Shouldnt' : 'should not',

                'shouldnt' : 'should not',

                'favourite' : 'favorite',

                'travelling' : 'traveling',

                'counselling' : 'counseling',

                'theatre' : 'theater',

                'cancelled' : 'canceled',

                'realized' : 'realised',

                'memorise' : 'memorize',

                'labour' : 'labor',

                'organisation':'organization',

                'wwii':'world war 2',

                'citicise':'criticize',

                'wwwyoutubecom' : 'youtube',

                'instagram': 'social medium',

                'whatsapp': 'social medium',

                'WeChat' : 'social medium',

                'snapchat': 'social medium',

                'Snapchat' : 'social medium',

                'Pinterest' : 'social medium',

                'bitcoins' : 'cryptocurrency',

                'bitcoin' : 'cryptocurrency',

                'Ethereum' : 'cryptocurrency',

                'cryptocurrencies' : 'cryptocurrency',

                'ethereum' : 'cryptocurrency',

                'Coinbase' : 'cryptocurrency',

                'Blockchain' : 'cryptocurrency',

                'Cryptocurrency' : 'cryptocurrency',

                'Litecoin' : 'cryptocurrency',

                'coinbase' : 'cryptocurrency',

                'altcoin' : 'cryptocurrency',

                'litecoin' : 'cryptocurrency',

                'cryptos' : 'cryptocurrency',

                'Fortnite' : 'game',

                'Nodejs' : 'programming',

                'nodejs' : 'programming',

                'ReactJS' : 'programming',

                'Golang' : 'programming',

                'counsellor' : 'counselling',

                'Tensorflow' : 'machine learning',

                'TensorFlow' : 'machine learning',

                'DeepMind' : 'machine learning',

                'Codeforces' : 'programming',

                'HackerRank' : 'programming',

                'CodeChef' : 'programming',

                'AngularJS' : 'programming',

                'PewDiePie' : 'gaming social medium',

                'brexit' : 'Brexit',

                'Xiaomi' : 'phone company',

                'OnePlus' : 'phone company',

                'mastrubation' : 'masturbation',

                'colour': 'color',

                'centre': 'center',

                'favourite': 'favorite',

                'travelling': 'traveling',

                'theatre': 'theater',

                'cancelled': 'canceled',

                'labour': 'labor',

                'organisation': 'organization',

                'wwii': 'world war 2',

                'citicise': 'criticize',

                'youtu ': 'youtube ',

                'Qoura': 'Quora',

                'sallary': 'salary',

                'Whta': 'What',

                'narcisist': 'narcissist',

                'howdo': 'how do',

                'whatare': 'what are',

                'howcan': 'how can',

                'howmuch': 'how much',

                'howmany': 'how many',

                'whydo': 'why do',

                'doI': 'do I',

                'theBest': 'the best',

                'howdoes': 'how does',

                'mastrubation': 'masturbation',

                'mastrubate': 'masturbate',

                "mastrubating": 'masturbating',

                'pennis': 'penis',

                'Etherium': 'cryptocurrency',

                'narcissit': 'narcissist',

                'bigdata': 'big data',

                '2k17': '2017',

                '2k18': '2018',

                'qouta': 'quota',

                'exboyfriend': 'ex boyfriend',

                'airhostess': 'air hostess',

                "whst": 'what',

                'watsapp': 'whatsapp',

                'demonitisation': 'demonetization',

                'demonitization': 'demonetization',

                'demonetisation': 'demonetization',

                'mofo': 'fuck',

                'ww2': 'world war 2',

                'havent': 'have not',

                'neonazis': 'Nazis',

                'hillary clinton': 'Hillary Clinton',

                'donald trump': 'Donald Trump'

               }







mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)

import string

string.printable

ascii_chars = string.printable

ascii_chars += " áéíóúàèìòùâêîôûäëïöüñõç"



#checks if a string of text contains any nonenglish characters (excluding punctuations, spanish, and french characters)

def contains_non_english(text):

    if all(char in ascii_chars for char in text):

        return 0

    else:

        return 1

    

#clean non english characters from string of text

def remove_non_english(text):

    return ''.join(filter(lambda x: x in ascii_chars, text))





def get_first_word(word):

    if(type(word) != "float"):

        return word.split(" ")[0]

    return "-1"



def get_cap_vs_length(row):

    if row["total_length"] == 0:

        return -1

    return float(row['capitals'])/float(row['total_length'])



def calc_max_word_len(sentence):

    maxLen = 0

    for word in sentence:

        maxLen = max(maxLen, len(word))

    return maxLen



#removes all single characters except for "I" and "a"

def remove_singles(text):

    return ' '.join( [w for w in text.split() if ((len(w)>1) or (w.lower() == "i") or (w.lower() == "a"))] )

    

#combines multiple whitespaces into single

def clean_text(x):

    x = str(x)

    x = x.replace("-", '')

    x = x.replace("/", '')

    x = x.replace("'",'')

    x = re.sub( '\s+', ' ', x).strip()

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
train = pd.read_csv("/kaggle/input/quoralb6/train.csv")

test = pd.read_csv("/kaggle/input/quoralb6/test.csv")

print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
# get number of bad words

def num_bad_words(text):

    badwords = 0

    input_words=text.split()

    for word in input_words:

        if word in bad_words:

            badwords += 1

    return badwords



# get number of good words

def num_good_words(text):

    goodwords = 0

    input_words=text.split()

    for word in input_words:

        if word in good_words:

            goodwords += 1

    return goodwords
#loads, generate features, then cleans



#Generate features

#df = pd.concat([train.loc[:, 'qid' : 'question_text'], test], sort = 'False')



print("--- Generating non_eng")

train["non_eng"] = train["question_text"].map(lambda x: contains_non_english(x))

test["non_eng"] = test["question_text"].map(lambda x: contains_non_english(x))

print("--- Generating first_word")

train["first_word"] = train["question_text"].map(lambda x: get_first_word(x))

test["first_word"] = test["question_text"].map(lambda x: get_first_word(x))

print("--- Generating total_length (num chars)")

train['total_length'] = train['question_text'].apply(len)

test['total_length'] = test['question_text'].apply(len)

print("--- Generating capitals")

train['capitals'] = train['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

test['capitals'] = test['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))



print("--- Generating caps_vs_length")

train['caps_vs_length'] = train.apply(lambda row: get_cap_vs_length(row),axis=1)

test['caps_vs_length'] = test.apply(lambda row: get_cap_vs_length(row),axis=1)



#print("--- Generating num_exclamation_marks")

#train['num_exclamation_marks'] = train['question_text'].apply(lambda comment: comment.count('!'))

#test['num_exclamation_marks'] = test['question_text'].apply(lambda comment: comment.count('!'))



print("--- Generating num_question_marks")

train['num_question_marks'] = train['question_text'].apply(lambda comment: comment.count('?'))

test['num_question_marks'] = test['question_text'].apply(lambda comment: comment.count('?'))



print("--- Generating num_punctuation")

train['num_punctuation'] = train['question_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))

test['num_punctuation'] = test['question_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))



#print("--- Generating num_symbols")

#train['num_symbols'] = train['question_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))

#test['num_symbols'] = test['question_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))



print("--- Generating num_words")

train['num_words'] = train['question_text'].apply(lambda comment: len(re.sub(r'[^\w\s]','',comment).split(" ")))

test['num_words'] = test['question_text'].apply(lambda comment: len(re.sub(r'[^\w\s]','',comment).split(" ")))



print("--- Generating num_unique_words")

train['num_unique_words'] = train['question_text'].apply(lambda comment: len(set(w for w in comment.split())))

test['num_unique_words'] = test['question_text'].apply(lambda comment: len(set(w for w in comment.split())))



print("--- Generating words_vs_unique")

train['words_vs_unique'] = train['num_unique_words'] / train['num_words']

test['words_vs_unique'] = test['num_unique_words'] / test['num_words']



#print("--- Generating num_smilies")

#train['num_smilies'] = train['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))

#test['num_smilies'] = test['question_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))



print("--- Generating num_sentences")

train['num_sentences'] = train['question_text'].apply(lambda comment: len(re.split(r'[.!?]+', comment)))

test['num_sentences'] = test['question_text'].apply(lambda comment: len(re.split(r'[.!?]+', comment)))



print("--- Generating max_word_len")

train['max_word_len'] = train['question_text'].apply(lambda comment: calc_max_word_len(re.sub(r'[^\w\s]','',comment).split(" ")))

test['max_word_len'] = test['question_text'].apply(lambda comment: calc_max_word_len(re.sub(r'[^\w\s]','',comment).split(" ")))



print("cleaning text")

train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))

test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))



print("remove single characters")

train["question_text"] = train["question_text"].apply(lambda x: remove_singles(x))

test["question_text"] = test["question_text"].apply(lambda x: remove_singles(x))



print("cleaning numbers")

train["question_text"] = train["question_text"].apply(lambda x: clean_numbers(x))

test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))



print("cleaning misspellings")

train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))

test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))



print("filling missing values")

#clean chinese, korean, japanese characters

print('cleaning characters')

train["question_text"] = train["question_text"].map(lambda x: remove_non_english(x))

test["question_text"] = test["question_text"].map(lambda x: remove_non_english(x))



## fill up the missing values

train["question_text"].fillna("").values

test["question_text"].fillna("").values



#for getting num good and bad words

from wordcloud import STOPWORDS

from collections import defaultdict

import operator



train1_df = train[train["target"]==1]

train0_df = train[train["target"]==0]



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



freq_dict_bad = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict_bad[word] += 1

freq_dict_bad = dict(freq_dict_bad)



freq_dict_good = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict_good[word] += 1

freq_dict_good = dict(freq_dict_good)



bad_words = sorted(freq_dict_bad, key=freq_dict_bad.get, reverse=True)[:1000]

good_words = sorted(freq_dict_good, key=freq_dict_good.get, reverse=True)[:1000]



print("--- Generating num_bad_words")

train["num_bad_words"] = train["question_text"].map(lambda x: num_bad_words(x))

test["num_bad_words"] = test["question_text"].map(lambda x: num_bad_words(x))



print("--- Generating num_good_words")

train["num_good_words"] = train["question_text"].map(lambda x: num_good_words(x))

test["num_good_words"] = test["question_text"].map(lambda x: num_good_words(x))

top_x = ['What', 'How', 'Why', 'Is', 'Can', 'Which', 'Do', 'If', 'Are', 'Who', 'I', 'Does', 'Where', 'Should', 'When', 'Will', "What's", 'In', 'Would', 'Have', 'Did', 'My', 'Has', 'As', 'Could', "I'm", 'Was', 'A', 'The', 'What’s', 'For', 'After', 'At', 'With', 'Am', 'Were', 'From', 'Since', 'To', 'On']

train["first_word"] = train["first_word"].map(lambda x: x if x in top_x else "Other")

test["first_word"] = test["first_word"].map(lambda x: x if x in top_x else "Other")

one_hot_encoded_first_word_train = pd.get_dummies(train["first_word"])

one_hot_encoded_first_word_test = pd.get_dummies(test["first_word"])



original_headers_train = list(train.columns.values)

original_headers_test = list(test.columns.values)

one_hot_encoded_first_word_headers_train = one_hot_encoded_first_word_train.columns.values

one_hot_encoded_first_word_headers_test = one_hot_encoded_first_word_test.columns.values

one_hot_encoded_first_word_headers_train = ["first_word_" + x for x in one_hot_encoded_first_word_headers_train]

one_hot_encoded_first_word_headers_test = ["first_word_" + x for x in one_hot_encoded_first_word_headers_test]

new_train_headers = original_headers_train + one_hot_encoded_first_word_headers_train

new_test_headers = original_headers_test + one_hot_encoded_first_word_headers_test



train_with_features = pd.concat([train, one_hot_encoded_first_word_test], axis=1, ignore_index=True)

train_with_features.columns = new_train_headers



test_with_features =  pd.concat([test, one_hot_encoded_first_word_test], axis=1, ignore_index=True)

test_with_features.columns = new_test_headers
#drop first word features that are insignificant

to_drop = [

       'first_word_Other', 'first_word_Can', 'first_word_Where',

       'first_word_Since', 'first_word_Did',

       'first_word_If', "first_word_What's", 'first_word_Should',

       'first_word_Who', 'first_word_I', 'first_word_Was',

       'first_word_The', 'first_word_When', 'first_word_Is',

       'first_word_Would', 'first_word_In', 'first_word_As',

       'first_word_Were', 'first_word_Will', 'first_word_What’s',

       'first_word_My', 'first_word_With', 'first_word_Am',

       'first_word_To', 'first_word_At', 'first_word_From',

       'first_word_Has', 'first_word_After',

       'first_word_Does', 'first_word_Could', 'first_word_A',

       "first_word_I'm", 'first_word_For', 'first_word_Have',

       'first_word_On']



train_with_features = train_with_features.drop(to_drop,axis=1)

test_with_features = test_with_features.drop(to_drop,axis=1)

#Tokenizes the data

def tokenize():



    ## fill up the missing values

    train_X = train["question_text"]

    test_X = test["question_text"]



    ## Tokenize the sentences

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    test_X = tokenizer.texts_to_sequences(test_X)



    ## Pad the sentences 

    train_X = pad_sequences(train_X, maxlen=maxlen)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    ## Get the target values

    train_y = train['target'].values

    

    #shuffling the data

    np.random.seed(2018)

    trn_idx = np.random.permutation(len(train_X))



    train_X = train_X[trn_idx]

    train_y = train_y[trn_idx]

    

    return train_X, test_X, train_y, tokenizer.word_index



'''def replace_words(sentence):

    new_sentence = ""

    for x in sentence.split(" "):

        if(x in bad_words):

            new_sentence += "bacon "

        else if(x in good_words):

            new_sentence += "eggs "

        else if(x in both_words):

            new_sentence += "ham "

        else:

            new_sentence += x + " "



def simple_tokenize():

    ## fill up the missing values

    train_X = train["question_text"]

    test_X = test["question_text"]

    

    print("removing bad / good words for train")

    train_X["question_text"] = train_X["question_text"].apply(lambda x: replace_words(x))

    print("removing bad / good words for test")

    test_X["question_text"] = test_X["question_text"].apply(lambda x: replace_words(x))



    ## Tokenize the sentences

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    print(train_X)

    test_X = tokenizer.texts_to_sequences(test_X)



    ## Pad the sentences 

    train_X = pad_sequences(train_X, maxlen=maxlen)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    ## Get the target values

    train_y = train['target'].values

    

    #shuffling the data

    np.random.seed(2018)

    trn_idx = np.random.permutation(len(train_X))



    train_X = train_X[trn_idx]

    train_y = train_y[trn_idx]

    

    return train_X, test_X, train_y, tokenizer.word_index '''
# Create our own embedding that take average of 3 embeddings as this is better than concatenating embeddings

train_X, test_X, train_y, word_index = tokenize()

embedding_matrix_1 = load_glove(word_index)

#unique_words2, embedding_matrix_2 = load_fasttext(word_index)

embedding_matrix_3 = load_para(word_index)



embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)

np.shape(embedding_matrix)


df = pd.concat([train_with_features.drop(['target'], axis=1), test], sort = 'False')

print(df.shape)

#TODO: REMOVE THIS LINE (tests on only first 300)

#df = df[:300]

#train = train[:300]

#train_X = train_X[:300]

#train_y = train_y[:300]

# len([ "total_length", "capitals","caps_vs_length", "num_exclamation_marks","num_question_marks","num_punctuation","num_symbols","num_words","num_unique_words","words_vs_unique","num_smilies"])

#Normalize feature values

features_names = df.drop(["qid","question_text","first_word"],axis=1).columns.values

tmp_df = df[features_names]



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

tmp_df = scaler.fit_transform(tmp_df)



train_data = tmp_df[0 : train.shape[0]]

test_data  = tmp_df[train.shape[0] : (train.shape[0] + test.shape[0])]





x_features_train = train_data

x_features_test  = test_data

pd.DataFrame(x_features_train).head()

#train_data.shape
#The Model itself

'''def model_lstm_atten(embedding_matrix):

    

    input_embeddings = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(input_embeddings)

    x = SpatialDropout1D(0.1)(x)

    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)

    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)

    

    atten_1 = Attention(maxlen)(x) # skip connect

    atten_2 = Attention(maxlen)(y)

    avg_pool = GlobalAveragePooling1D()(y)

    max_pool = GlobalMaxPooling1D()(y)

    

    input_features = Input(shape = (NUM_CUSTOM_FEATURES,))

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool, input_features])

    conc = Dense(16, activation="relu")(conc)

    conc = Dropout(0.1)(conc)

    outp = Dense(1, activation="sigmoid")(conc)    



    model = Model(inputs=[input_embeddings, input_features], outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

    

    return model'''

#The Model itself

def model_gru_cap(embedding_matrix):

    

    input_embeddings = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(input_embeddings)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNGRU(100, return_sequences=True, 

                                kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    

    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)

    x = Flatten()(x)

    

    '''atten_1 = Attention(maxlen)(x) # skip connect

    atten_2 = Attention(maxlen)(y)

    avg_pool = GlobalAveragePooling1D()(y)

    max_pool = GlobalMaxPooling1D()(y)'''

    

    input_features = Input(shape = (NUM_TOTAL_FEATURES,))

    #conc = concatenate([atten_1, atten_2, avg_pool, max_pool, input_features])

    #conc = Dense(16, activation="relu")(conc)

    #conc = Dropout(0.1)(conc)

    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)

    x = Dropout(0.12)(x)

    x = BatchNormalization()(x)

    

    outp = Dense(1, activation="sigmoid")(x)    



    model = Model(inputs=[input_embeddings, input_features], outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

    

    return model
# https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline



def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result
# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

#Attention and CyclicLR and Capsule



def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



# A Capsule Implement with Pure Keras

class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)



class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

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

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



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

        return input_shape[0],  self.features_dim



# https://www.kaggle.com/hireme/fun-api-keras-f1-metric-cyclical-learning-rate/code



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())

    



def f1(y_true, y_pred):

    '''

    metric from here 

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    '''

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

DATA_SPLIT_SEED = 2018

NUM_SPLITS = 4

clr = CyclicLR(base_lr=0.001, max_lr=0.002,

               step_size=300., mode='exp_range',

               gamma=0.99994)



train_meta = np.zeros(train_y.shape)

test_meta = np.zeros(test_X.shape[0])

splits = list(StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))

features_split = list(StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=DATA_SPLIT_SEED).split(x_features_train, train_y))
def train_pred(model, train_X, train_y, val_X, val_y, train_features, val_features, epochs=6, callback=None):

    for e in range(epochs):

        model.fit(x = [train_X, train_features], y = train_y, batch_size=512, epochs=1, validation_data=([val_X, val_features], val_y), callbacks = callback, verbose=0)

        pred_val_y = model.predict([val_X, val_features], batch_size=1024, verbose=0)

        

        

        '''print(pred_val_y)

        print(len(pred_val_y))

        print(val_y)

        print(len(val_y))

        print(set(val_y.flatten()) - set(pred_val_y.flatten()))

        print((pred_val_y > 0.33).astype(int))'''

        

        best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))

        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([test_X, x_features_test], batch_size=1024, verbose=0)

    print('=' * 60)

    return pred_val_y, pred_test_y, best_score
for ((train_idx, valid_idx), (train_f_idx, valid_f_idx)) in zip(splits, features_split):

        print(train_idx)

        print(valid_idx)

        print(train_f_idx)

        print(valid_f_idx)

        X_train = train_X[train_idx]

        y_train = train_y[train_idx]

        X_val = train_X[valid_idx]

        y_val = train_y[valid_idx]

        Xfeaturestrain = np.array(x_features_train)[train_f_idx]

        Xfeaturesval = np.array(x_features_train)[valid_f_idx]

        model = model_gru_cap(embedding_matrix)

        pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, Xfeaturestrain, Xfeaturesval, epochs = 6, callback = [clr,])

        print(pred_val_y.shape)

        print(pred_test_y.shape)

        train_meta[valid_idx] = pred_val_y.reshape(-1)

        test_meta += pred_test_y.reshape(-1) / len(splits)
threshold = threshold_search(train_y, train_meta)

threshold
f1_score(y_true=train_y, y_pred=train_meta > threshold["threshold"])
sub = pd.read_csv('../input/sample_submission.csv')

sub.prediction = (test_meta > threshold["threshold"]).astype(int)

sub.to_csv("submission.csv", index=False)