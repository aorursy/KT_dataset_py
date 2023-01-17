import numpy as np

np.random.seed(666)

import pandas as pd

from sklearn.cross_validation import train_test_split

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





#LOAD DATA

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")



#CREATE TARGET VARIABLE

train["EAP"] = (train.author=="EAP")*1

train["HPL"] = (train.author=="HPL")*1

train["MWS"] = (train.author=="MWS")*1

train.drop("author", 1, inplace=True)

target_vars = ["EAP", "HPL", "MWS"]

train.head(2)
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))

import string

## Number of words in the text ##

train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



num_vars = ["mean_word_len", "num_words_title", "num_punctuations", "num_chars"

            , "num_stopwords", "num_chars", "num_unique_words", "num_words"]
#STEMMING WORDS

import nltk.stem as stm

import re

stemmer = stm.SnowballStemmer("english")

train["stem_text"] = train.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))

test["stem_text"] = test.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))



#PROCESS TEXT: RAW

from keras.preprocessing.text import Tokenizer

tok_raw = Tokenizer()

tok_raw.fit_on_texts(train.text.str.lower())

tok_stem = Tokenizer()

tok_stem.fit_on_texts(train.stem_text)

train["seq_text_stem"] = tok_stem.texts_to_sequences(train.stem_text)

test["seq_text_stem"] = tok_stem.texts_to_sequences(test.stem_text)



#EXTRACT DATA FOR KERAS MODEL

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline



def get_keras_data(dataset, maxlen=20, scaler=None, tdfidf=None):

    if scaler==None:

        scaler = StandardScaler()

        scaler.fit(dataset[num_vars])

    if tdfidf==None:

        tdfidf = Pipeline(steps=[('tdfidf', TfidfVectorizer(analyzer='word', binary=False

                                , ngram_range=(1, 4), stop_words="english"))

                            , ('svd', TruncatedSVD(algorithm='randomized', n_components=20, n_iter=10,

                                       random_state=None, tol=0.0)

                                )])

        tdfidf.fit(dataset.text)

    X = {

        "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen),

        "num_input": scaler.transform(dataset[num_vars]),

        "svd_vect": tdfidf.transform(dataset.text)

    }

    return X, scaler, tdfidf





maxlen = 60

dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.9)

print("processing train...")

X_train, scaler, tdfidf = get_keras_data(dtrain, maxlen)

y_train = np.array(dtrain[target_vars])

print("processing valid...")

X_valid, _, _ = get_keras_data(dvalid, maxlen, scaler, tdfidf)

y_valid = np.array(dvalid[target_vars])

print("processing test...")

X_test, _, _ = get_keras_data(test, maxlen, scaler, tdfidf)



n_stem_seq = np.max( [np.max(X_valid["stem_input"]), np.max(X_train["stem_input"])])+1
#KERAS MODEL DEFINITION

from keras.layers import Dense, Dropout, Embedding

from keras.layers import Flatten, Input, SpatialDropout1D, Concatenate

from keras.models import Model

from keras.optimizers import Adam 

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



def get_callbacks(filepath, patience=2):

    es = EarlyStopping('val_loss', patience=patience, mode="min")

    msave = ModelCheckpoint(filepath, save_best_only=True)

    return [es, msave]



def get_model():

    embed_dim = 30

    dropout_rate = 0.9

    emb_dropout_rate = 0.9

   

    input_text = Input(shape=[maxlen], name="stem_input")

    

    input_num = Input(shape=[X_train["num_input"].shape[1]], name="num_input")

    

    input_svd = Input(shape=[X_train["svd_vect"].shape[1]], name="svd_vect")

    

    emb_lstm = SpatialDropout1D(emb_dropout_rate) (Embedding(n_stem_seq, embed_dim

                                                ,input_length = maxlen

                                                               ) (input_text))

    concatenate = Concatenate()([(Flatten() (emb_lstm)), input_num, input_svd])

    dense = Dropout(dropout_rate) (Dense(256) (concatenate))

    

    output = Dense(3, activation="softmax")(dense)



    model = Model([input_text, input_num, input_svd], output)



    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model



model = get_model()

model.summary()
#TRAIN KERAS MODEL

file_path = ".model_weights.hdf5"

callbacks = get_callbacks(filepath=file_path, patience=5)



model = get_model()

model.fit(X_train, y_train, epochs=150

          , validation_data=[X_valid, y_valid]

         , batch_size=512

         , callbacks = callbacks)
#MODEL EVALUATION

from sklearn.metrics import log_loss



model = get_model()

model.load_weights(file_path)



preds_train = model.predict(X_train)

preds_valid = model.predict(X_valid)



print(log_loss(y_train, preds_train))

print(log_loss(y_valid, preds_valid))
#PREDICTION

preds = pd.DataFrame(model.predict(X_test), columns=target_vars)

submission = pd.concat([test["id"],preds], 1)

submission.to_csv("./submission.csv", index=False)

submission.head()