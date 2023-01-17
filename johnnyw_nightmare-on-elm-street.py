import numpy as np

import pandas as pd

from collections import Counter

from random import shuffle, sample

import re

import itertools

import funcy

import keras

import keras.backend as k

from keras.layers import Dense, GlobalAveragePooling1D, Embedding

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier

import nltk

import nltk.data

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

from scipy.stats import mstats

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

% matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape, test.shape)
def content_words(text, target_tags):

    return [x[0] for x in nltk.pos_tag(text)

            if x[-1] in target_tags]



def extract_features(df):

    punct = re.compile('[\\.,\\?]')

    df["char_count"] = df["text"].str.len()

    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    df["unique_chars"] = df["text"].apply(lambda x: len(set([ch for ch in x])))

    df["av_word_len"] = df["char_count"] / df["word_count"]

    df["punct"] = df["text"].apply(lambda x: len(re.findall(punct, x)))

    df["punct_per_w"] = df["punct"] / df["word_count"]

    df["content_words"] = df["text"].apply(lambda x: content_words(x.split(), ['NN', 'NNS', 'JJ', 'RB', "VBG", "VBD", "VB"]))

    df["n_content_words"] = df["content_words"].apply(lambda x: len(x))

    df["cw_per_w"] = df["n_content_words"] / df["word_count"]

    df["adjectives"] = df["content_words"].apply(lambda x: content_words(x, ['JJ', "RB"]))

    df["n_adj"] = df["adjectives"].apply(lambda x: len(x))

    df["adj_per_w"] = df["n_adj"] / df["word_count"]

    return df
df = train.copy()

df = extract_features(df)
grouped = df.groupby("author")

features = ["word_count", "punct_per_w", "av_word_len", "char_count",

            "n_content_words", "cw_per_w", "n_adj", "adj_per_w", "unique_chars"]



for feature in features:

    print(feature)

    print(grouped[feature].agg(np.mean), "\n")
dfn = df[features].apply(lambda x: mstats.winsorize(x, limits=[0.05, 0.05]))

dfn.head()
scaler = StandardScaler()

pca = PCA(n_components=3)

X_reduced = pca.fit_transform(scaler.fit_transform(dfn.values))

d = {'EAP': 0, 'HPL': 1, 'MWS': 2}

df["author"] = df["author"].apply(lambda x: d[x])

y = to_categorical(df["author"].values)

xp = fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Blues, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])

plt.show()
pipe = Pipeline(

    steps=[

        ("scaler", StandardScaler()),

        ("poly_fs", PolynomialFeatures(degree=3)),

        ("pca", PCA(n_components=3))

    ])



X_poly_reduced = pipe.fit_transform(dfn[features].values)

xp = fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_poly_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,

           cmap=plt.cm.Blues, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])

plt.show()
df = df.drop(["id", "author", "text", "content_words", "adjectives"], axis=1)

skf = StratifiedKFold(n_splits=3)
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

X = dfn.values

for tr, te in skf.split(X, np.array([np.argmax(row) for row in y])):

    preds1 = clf.fit(X[tr], y[tr]).predict(X[te])

    print(metrics.log_loss(y[te], preds1))
df = train.copy()

dfte = test.copy()
def extract_symbols(df):

    """ separate symbols from words so they are vectorised independently"""

    print("extracting symbols")

    df2 = df.copy()

    df2["text"] = df["text"].apply(lambda x: "".join([" {} ".format(s)

                                                  if s in [",", ".", "#", "!", "'", "?",

                                                           "£", "$", "^", "&", "*", "(",

                                                          ")", "-", "+", "`", ":", ";"]

                                                  else "{}".format(s)

                                                  for s in list(x)])).apply(lambda x: re.sub(" +", " ", x))

    return df2



def remove_stopwords(df):

    print("removing stopwords")

    sws = set(stopwords.words('english'))

    df2 = df.copy()

    df2["text"] = df["text"].apply(lambda x: " ".join([w for w in x.split()

                                                       if w.lower() not in sws]))

    return df2



def _ngrams(text, n):

    words = text.split()

    return (words[i:i + n] for i in range(len(words) - n + 1))



def make_ngrams(df, n=3):

    print("making ngrams")

    df2 = df.copy()

    df2["ngrams"] = df["text"].apply(lambda x: " ".join(_ngrams(x, n)))

    return df2



def stemmer(df):

    print("stemming")

    pst = PorterStemmer()

    df2 = df.copy()

    df2["text"] = df["text"].apply(lambda x: " ".join([pst.stem(w) for w in x.split()]))

    return df2



def remove_rare(df, n=3):

    print("removing rare words...")

    # use the Counter class from python collections to calculate word frequencies

    word_counter = Counter(list(itertools.chain(*[w for w in df["text"].apply(lambda x: 

                                                                              [t.lower()

                                                                               for t in x.split()])])))

    rare_words = set([w for w in word_counter.keys() if word_counter[w] < n])

    print("removing {} words: {}...".format(len(rare_words), " ".join(list(rare_words)[:10])))

    df2 = df.copy()

    df2["text"] = df["text"].apply(lambda x: " ".join([w for w in x.split()

                                                  if w not in rare_words]))

    return df2



def trim_expand(df, minlen=15, maxlen=256):

    print("trimming / expanding")

    df2 = pd.DataFrame()

    df2["text"] = df["text"].apply(lambda x: x[maxlen:])

    df2["author"] = df["author"]

    dfx = pd.concat([df, df2])

    dfx = dfx[dfx["text"].map(len) > minlen]

    return dfx



def lower(df):

    df2 = df.copy()

    df2["text"] = df["text"].apply(lambda x: " ".join([w.lower() for w in x.split()]))

    return df2



def count_vectoriser(text, v=None):

    print("vectorising")

    if not v:

        v = CountVectorizer(stop_words=None, ngram_range=(1, 4))

        v.fit(text)

    else:

        print("using {}".format(v))

    return v.transform(text), v
# remember that the functions are applied in reverse order

text_prep_train = funcy.compose(remove_rare,

                                #stemmer,

                                #remove_stopwords,

                                #lower,

                                extract_symbols)



text_prep_test = funcy.compose(#stemmer,

                               #remove_stopwords,

                               #lower,

                               extract_symbols)
dftrain = text_prep_train(df)

dftest = text_prep_test(dfte)
print(dftrain.shape, dftest.shape)
d = {'EAP': 0, 'HPL': 1, 'MWS': 2}

dftrain["author"] = dftrain["author"].apply(lambda x: d[x])

y = to_categorical(dftrain["author"].values)
# y is now a one-hot encoding of authors

y[:4]
Xtrain, Xval, ytrain, yval = train_test_split(dftrain["text"].values, y)

[a.shape for a in (Xtrain, Xval, ytrain, yval)]
tokeniser = Tokenizer()

tokeniser.fit_on_texts(Xtrain)
print(np.mean([len(t) for t in Xtrain]))

print(np.median([len(t) for t in Xtrain]))

print(np.max([len(t) for t in Xtrain]))

print(len([t for t in Xtrain if len(t) > 300]) / len(Xtrain))

print(np.argmax([len(t) for t in Xtrain]))

# so let's use 300 as the max length
def tokenise(x, tokeniser, maxlen=256):

    return pad_sequences(

        sequences=tokeniser.texts_to_sequences(x),

        maxlen=maxlen)
X_train_tokens, X_val_tokens, X_test_tokens = (tokenise(x, tokeniser)

                                               for x in (Xtrain, Xval, dftest["text"].values))
# this is what the longest phrase looks like after being tokenised;

# shorted passages are padded with leading zeros

longest = np.argmax([len(t) for t in Xtrain])

X_train_tokens[longest]
input_dim = np.max(X_train_tokens) + 1

embedding_dims = 15

input_dim
def build_model(input_dims, embedding_dims=20, optimiser="adam"):

    model = Sequential()

    model.add(Embedding(input_dim=input_dims, output_dim=embedding_dims))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy",

                 optimizer=optimiser,

                 metrics=["accuracy"])

    return model
epochs = 50

model = build_model(input_dim, embedding_dims)
data = model.fit(X_train_tokens, ytrain, batch_size=16, validation_data=(X_val_tokens, yval),

                epochs=epochs, callbacks=[EarlyStopping(patience=2, monitor="val_loss")])
preds = model.predict_proba(X_val_tokens)
print(metrics.log_loss(yval, preds))

print(metrics.roc_auc_score(yval, preds))
# Doc2Vec works with sentences, so first define some functions to split text into sentences

def _tagger(sentence_n):

    sentence, i = sentence_n[1], sentence_n[0]

    return TaggedDocument(sentence.split(), [i])



def tag_sentences(text, tokenizer):

    tokens = tokenizer.tokenize(text)

    return list(map(_tagger, enumerate(tokens)))
df = train.copy()

dfte = test.copy()
Xtrain, Xval, ytrain, yval = train_test_split(dftrain["text"].values, y)

[a.shape for a in (Xtrain, Xval, ytrain, yval)]
# use the tagger to tag each phrase

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

docs = tag_sentences(" ".join(itertools.chain(Xtrain)), tokenizer)

print(docs[:5])

len(docs)
# now we can build the model and add its vocabulary

model = Doc2Vec(size=100, min_count=3, iter=1, window=8, workers=8)

model.build_vocab(docs)
model.corpus_count
# now we can train the model on the phrases in the training set

n = len(Xtrain)

i = 1

for sentence in Xtrain:

    if i % 500 == 0:

        print("trained on {}/{} phrases".format(i, n))

    doc = tag_sentences(sentence, tokenizer)

    # the loop below allows us to shuffle the words in the sentence after each epoch of training

    for _ in range(25):

        model.train(doc, total_examples=model.corpus_count, epochs=model.iter)

        shuffle(doc)

    i += 1

print("done: {}".format(model))

# save the trained model

with open("d2v.raw", "wb") as f:

    model.save(f)
model = Doc2Vec.load("d2v.raw")

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

model.wv.similar_by_vector("big")
# the model is now trained; we can use it to encode the phrases into vectors

vecs_tr = np.array([model.infer_vector(phrase) for phrase in Xtrain])

vecs_te = np.array([model.infer_vector(phrase) for phrase in Xval])

print(vecs_tr.shape, vecs_te.shape)
print(ytrain.shape, yval.shape)
clf = RandomForestClassifier(n_estimators=150, n_jobs=-1)

metrics.log_loss(yval, clf.fit(vecs_tr, ytrain).predict(vecs_te))
fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(vecs_tr)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=ytrain,

           cmap=plt.cm.Blues, edgecolor='k', s=40)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])

plt.show()