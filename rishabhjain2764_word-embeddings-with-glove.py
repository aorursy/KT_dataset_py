import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import operator
from nltk.corpus import stopwords
stopwords  = stopwords.words('english')
import re
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn import metrics 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('Train shape : ',train.shape[0])
print('Test shape : ',test.shape[0])
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embed_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
def build_vocab(texts):
    vocab = {}
    sentences = texts.apply(lambda x : x.split()).values
    for sentence in tqdm(sentences, disable = False):
        for word in sentence:
            vocab[word] = vocab.get(word, 0) + 1
    return vocab   
vocab  = build_vocab(train['question_text'])
print({k: vocab[k] for k in list(vocab)[:5]})
ei = set(embed_index.keys())
def check_coverage(vocab, embed_index):
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words = 0
    for word in tqdm(list(vocab.keys()), disable = False):
        if word in ei:
            known_words[word] = embed_index[word]
            num_known_words += vocab[word]
        elif word.lower() in ei:
            known_words[word.lower()] = embed_index[word.lower()]
            num_known_words += vocab[word]
        else:
            unknown_words[word] = vocab[word]
            num_unknown_words += vocab[word]
    print('{:.2%} of words of vocab are known.'.format(len(known_words)/len(vocab)))
    print('{:.2%} of all text is known.'.format(num_known_words/(num_known_words + num_unknown_words),2))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return known_words, unknown_words
kw, uw = check_coverage(vocab, embed_index)
uw[:10]
'?' in ei
train['question_text'] = train['question_text'].apply(lambda x : x.replace('?',''))
vocab  = build_vocab(train['question_text'])
print({k: vocab[k] for k in list(vocab)[:5]})
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
def clean_numbers(sentence):
    sentence = re.sub('[0-9]{5,}','#####', sentence)
    sentence = re.sub('[0-9]{4}','####', sentence)
    sentence = re.sub('[0-9]{3}','###', sentence)
    sentence = re.sub('[0-9]{2}','##', sentence)
    sentence = re.sub('[0-9]{1}','#', sentence)
    return sentence
    
train['question_text'] = train['question_text'].apply(lambda x : clean_numbers(x))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
print(uw[:20])
train['question_text'] = train['question_text'].apply(lambda x : x.replace("/"," "))
train['question_text'] = train['question_text'].apply(lambda x : x.replace("-"," "))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:30]
def clean_punctuations(sentence):
    sentence = str(sentence)
    for punct in '&':
        sentence = sentence.replace('&', f' {punct} ')
    for punct in '?!.,#$%\()*+-/:;<=>@[\\]^_{|}~"':
        sentence = sentence.replace(punct, '')
    return sentence
train['question_text'] = train['question_text'].apply(lambda x : clean_punctuations(x))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:30]
specials = ["’", "‘", "´", "`"]
def clean_apostrophe(sentence):
    for s in specials:
        sentence = sentence.replace(s,"'")
    return sentence    
train['question_text'] = train['question_text'].apply(lambda x : clean_apostrophe(x))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def remove_stopwords(sentence):
    words = sentence.split(" ")
    words = [contraction_mapping[word.lower()] if word.lower() in contraction_mapping.keys() else word for word in words]
    return " ".join(words)
train['question_text'] = train['question_text'].apply(lambda x : remove_stopwords(x))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:30]
train['question_text'] = train['question_text'].apply(lambda x: re.sub("'s", ' is', x))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
train['question_text'] = train['question_text'].apply(lambda x: x.replace('Quorans', 'Quora'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('Brexit', 'Europe'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('₹', 'rupee'))
'bible' in embed_index
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
'altcoin.com' in embed_index
train['question_text'] = train['question_text'].apply(lambda x: x.replace("'", ''))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('“', ''))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
train['question_text'] = train['question_text'].apply(lambda x: x.replace('cryptocurrencies', 'cryptocurrency'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('Redmi', 'mobile company'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('OnePlus', 'mobile company'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('Qoura', 'Quora'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('°C', 'degree celsius'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('etc…', 'etc.'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('Bhakts', 'supporter'))
train['question_text'] = train['question_text'].apply(lambda x: x.replace('bhakts', 'supporter'))
vocab  = build_vocab(train['question_text'])
kw, uw = check_coverage(vocab, embed_index)
uw[:20]
test['question_text'] = test['question_text'].apply(lambda x : x.replace("/"," "))
test['question_text'] = test['question_text'].apply(lambda x : x.replace("-"," "))

test['question_text'] = test['question_text'].apply(lambda x : clean_numbers(x))
test['question_text'] = test['question_text'].apply(lambda x : clean_punctuations(x))

test['question_text'] = test['question_text'].apply(lambda x : clean_apostrophe(x))

test['question_text'] = test['question_text'].apply(lambda x : remove_stopwords(x))

test['question_text'] = test['question_text'].apply(lambda x : re.sub("'s", ' is', x))

test['question_text'] = test['question_text'].apply(lambda x: x.replace("'", ''))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('“', ''))

test['question_text'] = test['question_text'].apply(lambda x: x.replace('Quorans', 'Quora'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('Brexit', 'Europe'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('₹', 'rupee'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('cryptocurrencies', 'cryptocurrency'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('Redmi', 'mobile company'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('OnePlus', 'mobile company'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('Qoura', 'Quora'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('°C', 'degree celsius'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('etc…', 'etc.'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('Bhakts', 'supporter'))
test['question_text'] = test['question_text'].apply(lambda x: x.replace('bhakts', 'supporter'))

X_train = train["question_text"].values
y = pd.read_csv('../input/train.csv')['target']
X_test = test["question_text"].values

x_train, x_val, y_train, y_val = train_test_split(X_train, y , test_size = 0.2)
all_embed = np.stack(embed_index.values())
print(all_embed.shape)
emb_mean, emb_std = all_embed.mean(), all_embed.std()
tokenizer = Tokenizer(num_words = 50000)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(X_test)
del train, test
del X_train, X_test
maxlen = 100
x_train = pad_sequences(x_train, maxlen = maxlen)
x_val = pad_sequences(x_val, maxlen = maxlen)
x_test = pad_sequences(x_test, maxlen = maxlen)
word_index = tokenizer.word_index
nb_words = min(50000, len(word_index))
print(nb_words)
embed_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 300))
for word, i in word_index.items():
    if i >= 50000: continue
    embed_vector = embed_index.get(word)
    if embed_vector is not None:
        embed_matrix[i] = embed_vector
def get_model():
    inp = Input(shape = (maxlen,))
    x = Embedding(50000, 300, weights = [embed_matrix])(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
model = get_model()

print(model.summary())
model.fit(x_train, y_train, batch_size = 512, epochs = 3, validation_data = (x_val, y_val), callbacks = [EarlyStopping(monitor='val_loss', min_delta = 0.0001)])
y_valpred = model.predict(x_val)
f1 = 0
threshold = 0

for thresh in np.arange(0.1, 0.501,0.01):
    f1score = np.round(metrics.f1_score(y_val, (y_valpred>thresh).astype(int)), 4)
    thresh = np.round(thresh,2)
    print('F1 score for threshold {} : {}'.format(thresh, f1score))
    if f1score > f1:
        f1 = f1score
        threshold = thresh
        #print('In {} : {}'.format(threshold,f1score))
print(threshold)
print(f1)
y_test = model.predict(x_test)
y_test = (y_test[:,0] > threshold).astype(np.int)
test = pd.read_csv("../input/test.csv")['qid']
submit_df = pd.DataFrame({"qid": test, "prediction": y_test})
submit_df.to_csv("submission.csv", index=False)