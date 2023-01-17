# Basic libraries

import warnings

import os

import re

import numpy as np

import random

import pandas as pd 

import json

from tqdm import tqdm



# Plotting

import matplotlib.pyplot as plt

%matplotlib inline



# Traditional Classifiers

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import LinearSVC



# Classifiers validation and test utils

from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_score

from sklearn import metrics as mt

from sklearn.multioutput import MultiOutputRegressor as mout

from sklearn.multioutput import MultiOutputClassifier as cout

from sklearn.metrics import roc_auc_score



# NLP utils

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import text_to_word_sequence



# Neural networks utils

from keras.callbacks import ModelCheckpoint

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Embedding

from keras.layers import LSTM, Bidirectional
#############################################

# INITIAL SETUP

###############

warnings.simplefilter("ignore", UserWarning)

seed = 7

np.random.seed(seed)



#############################################

# FILE CONSTANTS

###############

TRAIN_CSV_PATH = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"

TEST_CSV_PATH = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"

SAMPLE_SUBM_CSV_PATH = "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
def display_full(x):

    """Evita truncamento ao imprimir tabelas com texto longo"""

    pd.set_option('display.max_rows', len(x))

    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', 2000)

    pd.set_option('display.float_format', '{:20,.2f}'.format)

    pd.set_option('display.max_colwidth', -1)

    display(x)

    pd.reset_option('display.max_rows')

    pd.reset_option('display.max_columns')

    pd.reset_option('display.width')

    pd.reset_option('display.float_format')

    pd.reset_option('display.max_colwidth')



def hasNonASCII(s):

    """Checa se string possui algum caractere não padrão do inglês"""

    clean_str(s)

    try:

        s.encode(encoding='utf-8').decode('ascii')

    except UnicodeDecodeError:

        return True

    else:

        return False



def countNonASCII(s):

    """Conta quantos caracteres não padrão do inglês a string possui"""

    if hasNonASCII(s):

        space_split = s.split(' ')

        non_ascii_count = 0

        for item in space_split:

            if(hasNonASCII(item)):

                non_ascii_count += 1

        return non_ascii_count

    else:

        return 0
def clean_str(string):

    # split contractions like "he'll"

    string = re.sub(r"\'s", " \'s", string)

    string = re.sub(r"\'ve", " \'ve", string)

    string = re.sub(r"n\'t", " n\'t", string)

    string = re.sub(r"\'re", " \'re", string)

    string = re.sub(r"\'d", " \'d", string)

    string = re.sub(r"\'ll", " \'ll", string)

    

    # remove punctuation

    string = re.sub(r",", " , ", string)

    string = re.sub(r"!", " ! ", string)

    string = re.sub(r"\(", " \( ", string)

    string = re.sub(r"\)", " \) ", string)

    string = re.sub(r"\?", " \? ", string)

    string = re.sub(r"\s{2,}", " ", string) # "   "     spaces

    string = re.sub('<.*?>', '', string)    # <a src..> html tags

    string = re.sub(r'\d+', '', string)     # 1234      numbers

    string = re.sub("'", '', string)        # '         quotes

    string = re.sub(r'\W+', ' ', string)    # ABCD      abbrevs.

    string = string.replace('_', '')        # _         underscore

    

    # fix words like "finallllly" and "awwwwwesome"

    pttrn_repchar = re.compile(r"(.)\1{2,}")

    string = pttrn_repchar.sub(r"\1\1", string)



    # Emojis pattern

    emoji_pattern = re.compile("["

                u"\U0001F600-\U0001F64F"  # emoticons

                u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                u"\U0001F680-\U0001F6FF"  # transport & map symbols

                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251"

                u"\U0001f926-\U0001f937"u'\U00010000-\U0010ffff'u"\u200d"

                u"\u2640-\u2642"u"\u2600-\u2B55"u"\u23cf"u"\u23e9"u"\u231a"

                u"\u3030"u"\ufe0f"

    "]+", flags=re.UNICODE)

    string = emoji_pattern.sub(u'', string)



    # remove stop words (# words that don't add representativeness)

    # ex. "a", "at", "had", "has" ...

    stop_words = set(stopwords.words('english'))

    word_list = text_to_word_sequence(string)

    no_stop_words = [w for w in word_list if not w in stop_words]

    no_stop_words = " ".join(no_stop_words)

    string = no_stop_words

    

    # convert all letters to lower

    return string.strip().lower()
# Train Data

train_df_original = pd.read_csv(TRAIN_CSV_PATH)

train_df = train_df_original.copy()



# Labels of Train Data

Y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
# Test Data

test_df_original = pd.read_csv(TEST_CSV_PATH)

test_df = test_df_original.copy()
np.random.seed(17)

display_full(train_df.filter(["comment_text"]).sample(3)) # display_full avoids truncating long text
np.random.seed(17)

display_full(

    train_df

        .query('(toxic + severe_toxic + obscene + threat + insult + identity_hate)>0')

        .filter(["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])

        .sample(15)

) # display_full avoids truncating long text
train_df['comment_text'] = train_df['comment_text'].apply(clean_str)

train_df['nASCII'] = train_df['comment_text'].apply(hasNonASCII)

non_ascii_rows = train_df[train_df['nASCII']]

print("Samples with non-ASCII characters:", len(non_ascii_rows), "samples")
train_df['nASCII_count'] = (train_df['comment_text']

                                .apply(countNonASCII))
trainDF_nonASCII = train_df[train_df['nASCII']] # recreate df to include new column

nascii_tox_q = '(toxic + severe_toxic + obscene + threat + insult + identity_hate)>0'

tox_nonascii = (trainDF_nonASCII

                .query(nascii_tox_q)

                .sort_values(by=['nASCII_count'], ascending=False))

print("Toxic with non-ascii chars:", len(tox_nonascii))
display_full(tox_nonascii

               .filter(['id', 'comment_text']))
display_full(test_df_original.query("id == '51bb6805977dbbc2' | id == '564fa976421ed6a4'"))
train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_str(x))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_str(x))
models = [

   mout(LinearSVC()),

    cout(MultinomialNB()),

    mout(LogisticRegression(solver='lbfgs'))

]

model_names = ["LinearSVC", "MultinomialNB", "LogisticRegression"]

classes_name = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def train_norm(data, test_df):

    X = data['comment_text'].values

    count_vect = CountVectorizer()

    X_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()

    

    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    rkf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)



    model_name = []

    model_accuracy = []

    model_roc = []

    i = 0

    for model, name in zip (models, model_names):

        sample_submission = pd.read_csv(SAMPLE_SUBM_CSV_PATH)

        submission = sample_submission

        print("Modelo: ",name)

        for train, test in rkf.split(X):

            model.fit(X_tfidf[train], Y[train])

            y_pred = model.predict(X_tfidf[test])

            model_name.append(name)

            model_accuracy.append(mt.accuracy_score(Y[test], y_pred))

            model_roc.append(mt.roc_auc_score(Y[test], y_pred))    

        y_pred = model.predict(count_vect.transform(test_df.comment_text))     

        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred 

        submission.to_csv("submission_"+name+"_.csv", index=False)

        i = i+1

    disc_result = {'Model':  model_name, 'Accuracy': model_accuracy, "ROC": model_roc}      

    results = pd.DataFrame(disc_result, columns=["Model", "Accuracy", "ROC"])

    return results
results = train_norm(train_df, test_df)
results.head(30)
def plot(data):

    acc_linear = results.iloc[results["Model"].values == "LinearSVC"].Accuracy.values

    acc_nb = results.iloc[results["Model"].values == "MultinomialNB"].Accuracy.values

    acc_lg = results.iloc[results["Model"].values == "LogisticRegression"].Accuracy.values

    #loss = history.history['loss']

    #val_loss = history.history['val_loss']

    epochs = range(1, len(acc_nb) + 1)

    plt.plot(epochs, acc_linear, 'b', label='LinearSVC acc')

    plt.plot(epochs, acc_lg, 'b', label='LogisticRegression acc', color="g")

    plt.plot(epochs, acc_nb, 'b', label='MultinomialNB acc', color="r")

    plt.title('Grafíco com Acurácias')

    plt.legend(loc='best')

    

    roc_linear = results.iloc[results["Model"].values == "LinearSVC"].ROC.values

    roc_nb = results.iloc[results["Model"].values == "MultinomialNB"].ROC.values

    roc_lg = results.iloc[results["Model"].values == "LogisticRegression"].ROC.values



    plt.figure()

    plt.plot(epochs,  roc_linear, 'b', label='LinearSVC')

    plt.plot(epochs,  roc_nb, 'b', label='MultinomialNB', color="g" )

    plt.plot(epochs,  roc_lg, 'b', label='LogisticRegression', color="r")

    plt.title('Gráfico com o valor da curva ROC')

    plt.legend(loc='best')
plot(results)
gp = results.groupby(by="Model").Accuracy.mean()

gp
fig = plt.figure(figsize=(8,6))

plt.bar(["LinearSVC","LogisticRegression","MultinomialNB"], gp.values, color=["r", "g", "b"])

plt.xlabel("Labels")

plt.ylabel("Counts")

plt.title("Gráfico de barras para acurácia")

plt.show()
gp = results.groupby(by="Model").ROC.mean()

gp
fig = plt.figure(figsize=(8,6))

plt.bar(["LinearSVC","LogisticRegression","MultinomialNB"], gp.values, color=["r", "g", "b"])

plt.xlabel("Modelos")

plt.ylabel("ROC")

plt.title("Gráfico de barras para valores da curva ROC")

plt.show()
X = train_df['comment_text'].values

count_vect = CountVectorizer()

X_counts = count_vect.fit_transform(X)

tfidf_transformer = TfidfTransformer()

X_tfidf = tfidf_transformer.fit_transform(X_counts)



clf1, clf2, clf3 = models[0], models[1], models[2]

eclf = mout((VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='hard')))



for clf, label in zip([clf1, clf2, clf3, eclf], ['Linear Regression', 'Logisctic', 'Ensemble']):

    scores = cross_val_score(clf, X_tfidf, Y, cv=5, scoring='accuracy')    

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
def model():

    if pre_trained_wv is True:

        print("Usando glove..")

        num_words = min(max_fatures, len(word_index) + 1)

        weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)

        input_shape = (max_sequence_length,)

        model_input = Input(shape=input_shape, name="input", dtype='int32')    

        embedding = Embedding(

            num_words, 

            word_embedding_dim,

            input_length=max_sequence_length, 

            name="embedding", 

            weights=[weights_embedding_matrix], 

            trainable=False)(model_input)

        if bilstm is True:

            lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)

        else:

            lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    else:

        input_shape = (max_sequence_length,)

        model_input = Input(shape=input_shape, name="input", dtype='int32')    

        embedding = Embedding(max_fatures, embed_dim, input_length=max_sequence_length, name="embedding")(model_input)   

        if bilstm is True:

            lstm = Bidirectional(LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm"))(embedding)

        else:

            lstm = LSTM(embed_dim, dropout=0.2, recurrent_dropout=0.2, name="lstm")(embedding)

    

    model_output = Dense(6, activation='sigmoid', name="sigmoid")(lstm)

    model = Model(inputs=model_input,outputs=model_output)

    return model
def train(model, X_train, Y_train, X_val, Y_val):

    filepath="weights.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [checkpoint]



    history = model.fit(

        X_train, 

        Y_train, 

        validation_data=(X_val, Y_val),

        epochs=30,

        batch_size=3000, 

        shuffle=True,

        verbose=1, callbacks=callbacks_list)  



    # Plot

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show()

    return model
word_embedding_dim = 50     # pre-trained word embedding dimension

batch_size = 300            # number of samples to be used on each gradient update

max_fatures = 5000          # maximum amount of words to keep in the vocabulary

embed_dim = 128             # output embedding layer dimension

max_sequence_length = 300   # maximum sentence length is limited to 300 words

bilstm = False

pre_trained_wv = False





def prepare_data(data, label = None, test=False):    

    text = []

    for row in data['comment_text'].values:

        text.append(row)



    tokenizer = Tokenizer(num_words=max_fatures, split=' ')

    tokenizer.fit_on_texts(text)

    X = tokenizer.texts_to_sequences(text)  



    X = pad_sequences(X, maxlen=max_sequence_length)

    #X = pad_sequences(X)



    word_index = tokenizer.word_index

    #Y = pd.get_dummies(data[label]).values



    if test == True:

        return X, word_index, tokenizer

    else:

        #Y = pd.get_dummies(data[labels]).values

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

    

        X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.10, random_state = 42)

        return X_train, X_test, Y_train, Y_test, word_index, X_val, Y_val, tokenizer





def load_pre_trained_wv(word_index, num_words, word_embedding_dim):

    embeddings_index = {}

    f = open(os.path.join('../input/glove6b50dtxt', 'glove.6B.{}d.txt'.format(word_embedding_dim)), encoding='utf-8')

    for line in tqdm(f):

        values = line.rstrip().rsplit(' ')

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

    f.close()



    print('%s word vectors.' % len(embeddings_index))



    embedding_matrix = np.zeros((num_words, word_embedding_dim))

    for word, i in word_index.items():

        if i >= max_fatures:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector



    return embedding_matrix
def test(X_test, model):

    X_test_data = tokenizer.texts_to_sequences(X_test['comment_text'])

    X_test_data = pad_sequences(X_test_data, maxlen=max_sequence_length, dtype='int32', value=0)

    y_pred = model.predict(X_test_data, batch_size=1000, verbose=1)



    return y_pred
X_train, X_test, Y_train, Y_test, word_index, X_val, Y_val, tokenizer = prepare_data(train_df)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
pre_trained_wv = False

model1 = model()

model1.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

print(model1.summary())

model1 = train(model1, X_train, Y_train, X_val, Y_val)   

y_pred = model1.predict(X_test,batch_size=1024,verbose=6)

print("AUC: ", roc_auc_score(Y_test, y_pred))
y_pred = test(test_df, model1)

submission = sample_submission

submission['id'] = test_df['id']

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission_lstm.csv', index=False)
pre_trained_wv = True

model_glove = model()

model_glove.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

print(model_glove.summary())

model = train(model_glove, X_train, Y_train, X_val, Y_val)    

y_pred = model_glove.predict(X_test,batch_size=1024,verbose=6)

print("AUC: ", roc_auc_score(Y_test, y_pred))
def test(X_test, model):

    X_test_data = tokenizer.texts_to_sequences(X_test['comment_text'])

    X_test_data = pad_sequences(X_test_data, maxlen=max_sequence_length, dtype='int32', value=0)

    y_pred = model.predict(X_test_data, batch_size=1000, verbose=1)



    return y_pred
y_pred = test(test_df, model_glove)
sample_submission = pd.read_csv(SAMPLE_SUBM_CSV_PATH)

submission = sample_submission

submission['id'] = test_df['id']

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred

submission.to_csv('submission_lstm_glove,.csv', index=False)
submission.head()