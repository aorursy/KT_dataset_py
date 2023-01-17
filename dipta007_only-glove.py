# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# add progress bar

# !pip install tqdm --upgrade

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

tqdm.pandas()
# Global Constants

emb_size = 300

max_features = 200000

maxlen = 100
def clean_memory(*args):

    for arg in args:

        del arg

    import gc

    gc.collect()

    time.sleep(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



def plot_it(x, y, data):

    sns.boxplot(x=x, y=y, data=data)

#     plt.show()
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



path = '/kaggle/input/quora-insincere-questions-classification'

# path = '/content'

train_df = pd.read_csv(path + "/train.csv")

test_df = pd.read_csv(path + "/test.csv")



train_df["question_text"].fillna("_na_", inplace=True)

test_df["question_text"].fillna("_na_", inplace=True)
train_df["num_words"] = train_df["question_text"].progress_apply(lambda x: len(str(x).split()))

train_df["num_unique_words"] = train_df["question_text"].progress_apply(lambda x: len(set(str(x).split())))

train_df["num_chars"] = train_df["question_text"].progress_apply(lambda x: len(str(x)))

# stat_data["num_words_upper"] = train_df["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# stat_data["num_words_title"] = train_df["question_text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# stat_data["mean_word_len"] = train_df["question_text"].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["num_words"] = test_df["question_text"].progress_apply(lambda x: len(str(x).split()))

test_df["num_unique_words"] = test_df["question_text"].progress_apply(lambda x: len(set(str(x).split())))

test_df["num_chars"] = test_df["question_text"].progress_apply(lambda x: len(str(x)))
# plot_it('target', 'num_words', stat_data)
# plot_it('target', 'num_unique_words', stat_data)
# plot_it('target', 'num_chars', stat_data)
# plot_it('target', 'num_words_upper', stat_data)
# plot_it('target', 'num_words_title', stat_data)
# plot_it('target', 'mean_word_len', stat_data)
# Vocabulary build with count

def build_vocab(sentences, verbose =  True):

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
emb_path = '/kaggle/input/quora-insincere-questions-classification/embeddings'



def load_all():

    word2vec_format = {}

    glove = [o.split(" ")[0] for o in tqdm(open(emb_path + '/glove.840B.300d/glove.840B.300d.txt'))]

    for word in tqdm(glove):

        word2vec_format[word] = 1

    clean_memory(glove)



#     paragram = [o.split(" ")[0] for o in open(emb_path + '/paragram_300_sl999/paragram_300_sl999.txt', encoding="utf8", errors='ignore') if len(o)>100]

#     para = 0

#     for word in tqdm(paragram):

#         word2vec_format[word] = 1

#     clean_memory(paragram)

    

    return word2vec_format



emb_all = load_all()
# Coverage check for current embedding

import operator 



def check_coverage(vocab, embeddings_index):

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
sentences = train_df["question_text"].progress_apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab, emb_all)

oov[:10]
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }



def clean_special_chars(text, punct=punct, mapping=punct_mapping):

    for p in mapping:

        if p in text:

            text = text.replace(p, mapping[p])

    

    for p in punct:

        if p in text:

            text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        if s in text:

            text = text.replace(s, specials[s])

    

    return text
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_special_chars(x))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: clean_special_chars(x))

sentences = train_df["question_text"].progress_apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab,emb_all)

oov[:10]
# contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }



# def clean_contractions(text, mapping):

#     specials = ["’", "‘", "´", "`"]

#     for s in specials:

#         text = text.replace(s, "'")

#     text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

#     return text
# train["question_text"] = train["question_text"].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# sentences = train["question_text"].apply(lambda x: x.split())

# vocab = build_vocab(sentences)

# oov = check_coverage(vocab,glove_emb)

# oov[:10]
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}



def correct_spelling(x, dic):

    for word in dic.keys():

        if word in x:

            x = x.replace(word, dic[word])

    return x
train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: correct_spelling(x, mispell_dict))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: correct_spelling(x, mispell_dict))

sentences = train_df["question_text"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab, emb_all)

oov[:10]
go_to_more_common_words = {

    'Redmi': 'Mobile',

    'OnePlus': 'Mobile',

    'Quorans': 'Quoran',

    'cryptocurrencies': 'technology',

    'Cryptocurrency': 'Technology',

    'Blockchain': 'Technology',

    'Upwork': 'Technology',

    'HackerRank': 'Programming',

}



train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: correct_spelling(x, go_to_more_common_words))

test_df["question_text"] = test_df["question_text"].progress_apply(lambda x: correct_spelling(x, go_to_more_common_words))

sentences = train_df["question_text"].apply(lambda x: x.split())

vocab = build_vocab(sentences)

oov = check_coverage(vocab, emb_all)

oov[:10]
clean_memory(oov, vocab, sentences, mispell_dict, go_to_more_common_words, punct, punct_mapping, emb_all)
def load_all_emb(word_index):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    glove = dict(get_coefs(*o.split(" ")) for o in tqdm(open(emb_path + '/glove.840B.300d/glove.840B.300d.txt')) if o.split(" ")[0] in word_index)

#     paragram = dict(get_coefs(*o.split(" ")) for o in open(emb_path + '/paragram_300_sl999/paragram_300_sl999.txt', encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

    

    global max_features

    all_embs = np.stack(glove.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

#     emb_mean, emb_std = -0.005838499, 0.48782197

    clean_memory(all_embs)



    nb_words = min(max_features, len(word_index))

    print(nb_words)

    max_features = nb_words

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, emb_size))

    for word, i in tqdm(word_index.items()):

        if i >= nb_words: continue

        glove_vector = glove.get(word)

#         paragram_vector = paragram.get(word)

        paragram_vector = None

        if glove_vector is not None and paragram_vector is not None:

            embedding_matrix[i] = 0.7 * glove_vector + 0.3 * paragram_vector

        elif glove_vector is not None:

            embedding_matrix[i] = glove_vector

        elif paragram_vector is not None:

            embedding_matrix[i] = paragram_vector

    clean_memory(glove_vector, paragram_vector, glove)

            

    return embedding_matrix
emb_path = '/kaggle/input/quora-insincere-questions-classification/embeddings'

# emb_path = '/content'

tokenizer = Tokenizer(num_words=max_features, filters="")

tokenizer.fit_on_texts(np.concatenate((train_df["question_text"].values, test_df["question_text"].values)))

embeddings = load_all_emb(tokenizer.word_index)



print(max_features)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=47)



## fill up the missing values

train_X = train_df["question_text"].values

val_X = val_df["question_text"].values

test_X = test_df["question_text"].values



## 2nd type input

train_X2 = train_df[['num_words', 'num_unique_words', 'num_chars']]

val_X2 = val_df[['num_words', 'num_unique_words', 'num_chars']]

test_X2 = test_df[['num_words', 'num_unique_words', 'num_chars']]



## Tokenize the sentences

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



print('Train shape: {}'.format(train_X.shape))

print('Validation shape: {}'.format(val_X.shape))

print('Test shape: {}'.format(test_X.shape))



clean_memory(train_df, val_df, test_df)
print("Train shape: {}".format(train_X.shape))

print("Validation shape: {}".format(val_X.shape))
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor="val_f1", patience=2, restore_best_weights=True, mode="max")

reduce_lr = ReduceLROnPlateau(monitor="val_f1", mode="max")

callbacks = [early_stopping, reduce_lr]
from keras import backend as K



def f1(y_true, y_pred):

    '''

    metric from here 

    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    '''

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
train_X2.shape, train_X.shape, train_y.shape

# train_X2.head()
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Input, Embedding, LSTM, Reshape, Flatten, BatchNormalization, GlobalAveragePooling1D, SpatialDropout1D, CuDNNGRU, CuDNNLSTM, Dense, Bidirectional, GlobalMaxPooling1D, Dropout, Conv1D, concatenate

from tensorflow.python.keras.models import Model



# 1st Input

inp1 = Input(shape=(maxlen, ))

x = Embedding(max_features, emb_size, weights=[embeddings], trainable=False)(inp1)

x = SpatialDropout1D(0.4)(x)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

# x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)

x = Conv1D(64, 1)(x)



# avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)



# 2nd Input

inp2 = Input(shape=(3, ))

y = Dense(64)(inp2)



xy = concatenate([max_pool, y], axis=1)

# xy = Flatten()(xy)

xy = Dense(128, activation="relu")(xy)

xy = Dropout(0.1)(xy)

xy = BatchNormalization()(xy)

outp = Dense(1, activation="sigmoid")(xy)



model = Model(inputs=[inp1, inp2], outputs=outp)



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
model.summary()
model.fit([train_X, train_X2], train_y, batch_size=512, epochs=20, validation_data=([val_X, val_X2], val_y), callbacks=callbacks, verbose=True)
clean_memory(train_X, train_y)
# from sklearn.metrics import f1_score

# optimal = -100

# optimal_point = 0

# pred_val_y = model.predict(test_val_x)

# for thresh in np.arange(0.1, 0.501, 0.01):

#     thresh = np.round(thresh, 2)

#     now_f1 = f1_score(test_val_y, (pred_val_y > thresh).astype(int))

#     if now_f1 > optimal:

#       optimal_point = thresh

#       optimal = now_f1

#     print("F1 score at threshold {0} is {1}".format(thresh, now_f1))
# print("Optimal F1 {} on threshhold {}".format(optimal, optimal_point))
pred_val_y = model.predict([val_X, val_X2], batch_size=512, verbose=1)



def scoring(y_true, y_proba, verbose=True):

    from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

    from sklearn.model_selection import RepeatedStratifiedKFold



    def threshold_search(y_true, y_proba):

        precision , recall, thresholds = precision_recall_curve(y_true, y_proba)

        thresholds = np.append(thresholds, 1.001) 

        F = 2 / (1/precision + 1/recall)

        best_score = np.max(F)

        best_th = thresholds[np.argmax(F)]

        return best_th 





    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)



    scores = []

    ths = []

    for train_index, test_index in rkf.split(y_true, y_true):

        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]

        y_true_train, y_true_test = y_true[train_index], y_true[test_index]



        # determine best threshold on 'train' part 

        best_threshold = threshold_search(y_true_train, y_prob_train)



        # use this threshold on 'test' part for score 

        sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))

        scores.append(sc)

        ths.append(best_threshold)



    best_th = np.mean(ths)

    score = np.mean(scores)



    if verbose: print(f'Best threshold: {np.round(best_th, 4)}, Score: {np.round(score,5)}')



    return best_th, score



optimal_point1, optimal1 = scoring(val_y, pred_val_y)

print("Optimal F1 {} on threshhold {}".format(optimal1, optimal_point1))
clean_memory(val_X, val_y, pred_val_y)
test_df.head()
all_preds = model.predict([test_X, test_X2], batch_size=512, verbose=1)

pred_test_y = (np.array(all_preds) > optimal_point1).astype(np.int)
pred_test_y1 = np.asarray([y[0] for y in pred_test_y])

pred_test_y1.shape

pred_test_y1[:10]
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": pred_test_y1})

submit_df.to_csv("submission.csv", index=False)