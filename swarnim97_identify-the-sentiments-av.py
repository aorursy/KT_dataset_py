!pip install tqdm==4.33.0
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tqdm import tqdm

tqdm.pandas()

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/identify-the-sentiments-analytics-vidhya/train.csv')

test = pd.read_csv('../input/identify-the-sentiments-analytics-vidhya/test.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



def load_embeddings(embed_dir=EMBEDDING_PATH):

    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))

    return embedding_index
embeddings_index = load_embeddings()
print('Found %s word vectors.' % len(embeddings_index))
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
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
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

    print("Total words common in both vocabulary and in embeddings_index",len(a))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
'insta' in embeddings_index
train['tweet'][0]
contraction_mapping = {"Won't": "will not", "1950's": "1950s", "1983's": "1983", "ain't": "is not", "aren't": "are not", "Bretzing's": "", "Bundycon's": "Bundycon", "Can't": "cannot", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "C'mon": "Come on", "Denzel's": "Denzel", "didn't": "did not",  "doesn't": "does not", "Don't": "Do not", "don't": "do not", "Farmer's": "Farmers", "FBI's": "FBI", "Ferguson's": "Ferguson", "Hammond's": "Hammond", "hadn't": "had not", "hasn't": "has not", "Haven't": "Have not", "haven't": "have not", "he'd": "he would", "Here's": "Here is", "here's": "here is","he'll": "he will", "he's": "he is", "He's": "He is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "I'd": "I had", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "I'm": "I am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "It's": "it is", "Kay's": "Kay", "let's": "let us", "Let's": "let us", "ma'am": "madam", "mayn't": "may not", "Medford's": "Medford", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "Murphy's": "Murphys", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "Paula's": "Paula", "Portland's": "Portlands", "Portlander's": "Portlanders", "publication's": "publications", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "She's": "She is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "Tastebud's": "Tastebuds", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "That's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "There's": "There is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "They're": "They are", "they've": "they have", "to've": "to have", "Trump's": "trump is", "U.S.": "United state", "U.S": "United state", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "We'll": "We will", "we'll've": "we will have", "Wendy's": "Wendy", "we're": "we are", "We're": "We are", "we've": "we have", "We've": "We have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "What's": "What is",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "Who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "Wouldn't": "Would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "You'd": "You had","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "You're": "you are", "you've": "you have", "Zoo's": "zoos", "zoo's": "zoos" }
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
print(known_contractions(embeddings_index))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train['tweet'] = train['tweet'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

test['tweet'] = test['tweet'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train['tweet'][3]
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
'givewebsite' in embeddings_index
mispell_dict = {'ShoppingList': 'shopping list', 'follow4follow': '', 'f4f': '', 'likeforlike': '', 'giveawaywebsite': 'give away website', 'snapspeed': 'app', 'instacool': '', 'bestoftheday': 'goodday', 'igdaily': '', 'instahub': '', 'instalike': '', 'iphoneplus': 'iphone', 'instago': '', 'instapic': '', 'tweegram': '', 'instadaily': 'daily', 'instamood': 'mood', 'iphoneonly': 'iphone', 'iphonesia': '', 'iphonesia': '', 'instagood': 'insta', 'xperiaZ': 'xperia', 'iphoneplus': 'iPhone', 'iPhone8': 'iPhone', 'iphoneX': 'iphone', 'iphone5s': 'iphone', 'applewatch': 'apple watch', 'sonyalpha': 'sony alpha', 'applesucks': 'apple sucks', 'ios7': 'ios', 'ipadmini': 'ipad mini', 'sonyphotos': 'sony photos', 'FuckYou': 'Fuck You', 'shotoniphone': 'shot on iphone', 'giveawaywebsite': 'give away website', 'iPhone6': 'iphone', 'oneplus': 'smartphone', 'unitedState': 'united State', 'sonyphoto': 'sony photo', 'iphoneplus': 'iphone plus', 'sonylens': 'sony lens', 'iphone8': 'iphone', 'sonyphotography': 'sony photography', '$&@*#': '', '#$&@*#': '', '#': '', '$&': '', '#$&': '', 'hateapple': 'hate apple', 'Андроид': '', 'ReallyReal': 'Really Real', 'giveawaywebsite': 'give away website', 'FollowSunday': 'Follow Sunday', 'TeamFollowback': 'Team Follow back', 'phonecase': 'phone case', 'iphone7': 'iphone', 'newphone': 'new phone', 'sougofollow': '', 'iphonex': 'iphone', 'iphone6': 'iphone', 'iPhoneX': 'iphone', 'FOLLOWBACK': 'FOLLOW BACK'}
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
train['tweet'] = train['tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

test['tweet'] = test['tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
train['tweet'][0]
train['tweet'][11]
train['tweet'][7]
def clean_text(text):

    text = text.replace('\n', ' ')

    result = re.sub(r"http\S+", "website", text)

    result = re.sub(r"https\S+", "website", result)

    result = re.sub(r"pic\S+", "website", result)

    result = re.sub(r"@\S+", "", result)

    return result
train['tweet'] = train['tweet'].apply(lambda x: clean_text(x))

test['tweet'] = test['tweet'].apply(lambda x: clean_text(x))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
train['tweet'][0]
train['tweet'][11]
train['tweet'][7]
import emoji

print(emoji.demojize('Python is 😊'))
def emoji_cleaning(text):

    text = emoji.demojize(text)

    return text
train['tweet'] = train['tweet'].apply(lambda x: emoji_cleaning(x))

test['tweet'] = test['tweet'].apply(lambda x: emoji_cleaning(x))
train['tweet'][:5]
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
print(unknown_punct(embeddings_index, punct))
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
train['tweet'] = train['tweet'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test['tweet'] = test['tweet'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))
train_sentences = train["tweet"].progress_apply(lambda x: x.split()).values

test_sentences = test["tweet"].progress_apply(lambda x: x.split()).values

train_vocab = build_vocab(train_sentences)

test_vocab = build_vocab(test_sentences)

print({k: train_vocab[k] for k in list(train_vocab)[:5]})

print({k: test_vocab[k] for k in list(test_vocab)[:5]})
print(len(train_vocab))

print(len(test_vocab))
train_oov = check_coverage(train_vocab,embeddings_index)

test_oov = check_coverage(test_vocab,embeddings_index)
train_oov[:10]
test_oov[:10]
train['tweet'][:5]
'BACK' in embeddings_index
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
train.head()
train_df, validate_df = train_test_split(train, test_size=0.05, stratify=train['label'])
tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(list(train['tweet']) +list(test['tweet']))
print(train_df.shape)

print(validate_df.shape)
word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))
max_length = max([len(s.split()) for s in list(train['tweet'])])
print(max_length)
max_length = 40
X_train_pad = tokenizer_obj.texts_to_sequences(train_df['tweet'])

y_train = train_df['label'].values

X_test_pad = tokenizer_obj.texts_to_sequences(validate_df['tweet'])

y_test = validate_df['label'].values
x_test = test['tweet'].fillna('').values

test_sequences = tokenizer_obj.texts_to_sequences(x_test)

Test_pad = pad_sequences(test_sequences, maxlen=max_length)

print(Test_pad.shape)
len(X_train_pad)
X_train_pad = pad_sequences(X_train_pad, maxlen=max_length)

X_test_pad = pad_sequences(X_test_pad, maxlen=max_length)
EMBEDDING_DIM = 300
nb_words = len(word_index)
print('shape of X_train_pad tensor:', X_train_pad.shape)

print('shape of y_train tensor:', y_train.shape)

print('shape pf X_test_pad tensor:', X_test_pad.shape)

print('shape of y_test tensor:', y_test.shape)
def build_embedding_matrix(word_index, embeddings_index, max_features, lower = True, verbose = True):

    embedding_matrix = np.zeros((max_features, 300))

    for word, i in tqdm(word_index.items(),disable = not verbose):

        if lower:

            word = word.lower()

        if i >= max_features: continue

        try:

            embedding_vector = embeddings_index[word]

        except:

            embedding_vector = embeddings_index["unknown"]

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return embedding_matrix
embedding_matrix = build_embedding_matrix(word_index, embeddings_index, nb_words)
train.info()
print(train_df.shape)

print(validate_df.shape)
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, CuDNNLSTM, SpatialDropout1D, concatenate, Input

from keras.layers import Bidirectional, BatchNormalization

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.models import Model

from keras.layers import Dense, Dropout, Activation

from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
inp=Input(shape=(max_length,))

x = Embedding(nb_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),input_length=max_length, trainable=True)(inp)

x = Bidirectional(CuDNNLSTM(EMBEDDING_DIM, return_sequences=False))(x)

x = Dense(100, kernel_initializer='normal', activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(50, kernel_initializer='normal', activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inp], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train_pad, y_train, epochs=4, validation_data=(X_test_pad, y_test))
from sklearn import metrics
val_y = model.predict(X_test_pad, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_test, (val_y>thresh).astype(int))))
metrics.roc_auc_score(y_test, val_y)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.show()
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



plt.clf()

acc = history_dict['acc']

val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
import pandas

vect_range=list(range(100))

range_epoch = pandas.DataFrame(vect_range)

val_acc2 = pandas.DataFrame(val_acc)

best_acc = pandas.concat([range_epoch, val_acc2], axis=1)

best_acc.columns = ['a','b']

epoch=best_acc.loc[best_acc['b']==max(best_acc['b']),"a"]



new_epoch=int(epoch+1)

print(int(new_epoch))
val_y = (val_y > 0.38).astype(int)
metrics.accuracy_score(y_test, val_y)
metrics.confusion_matrix(y_test, val_y)
y_pred = model.predict(Test_pad)
y_pred = (y_pred > 0.38).astype(int)
y_pred
final = pd.read_csv('../input/identify-the-sentiments-analytics-vidhya/sample_submission.csv')

final['label'] = y_pred
final.to_csv('submission.csv', index=False)
final.head()