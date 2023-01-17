# Scikit Learn

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve



from sklearn import model_selection, preprocessing, metrics, ensemble

from keras.preprocessing.sequence import pad_sequences

import numpy as np

import pandas as pd

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler



import os



from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Flatten, MaxPooling1D

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import gc



print("Libraries loaded")
print(os.listdir('../input/jigsaw-unintended-bias-in-toxicity-classification'))
train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")



# train_df =   train_df.head(1000)

# test_df =   test_df.head(1000)

# Train_1 = train[train['target']==1]

# Train_0 = train[train['target']==0]

# train =  pd.concat([Train_1,Train_0.head(len(Train_1))], axis=0)
# train[train['target']==1].head(10)
# train['target'] = np.where(train['target'] >= 0.5, 1, 0)

# train["target"].value_counts()



NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)





def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix



for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)
# stopwords = set(STOPWORDS)





# wordcloud = WordCloud(

#                           background_color='white',

#                           stopwords=stopwords,

#                           max_words=200,

#                           max_font_size=40, 

#                           random_state=42

#                          ).generate(str(train[train.target==0]['comment_text']))



# print(wordcloud)

# fig = plt.figure(1)

# plt.imshow(wordcloud)

# plt.axis('off')

# plt.show()

# fig.savefig("word1.png", dpi=900)
# stopwords = set(STOPWORDS)



# wordcloud = WordCloud(

#                           background_color='white',

#                           stopwords=stopwords,

#                           max_words=200,

#                           max_font_size=40, 

#                           random_state=42

#                          ).generate(str(train[train.target==1]['comment_text']))



# print(wordcloud)

# fig = plt.figure(1)

# plt.imshow(wordcloud)

# plt.axis('off')

# plt.show()

# fig.savefig("word1.png", dpi=900)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 50 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a question to use



## fill up the missing values

train_X = train_df["comment_text"].fillna("_na_").values

val_X = val_df["comment_text"].fillna("_na_").values

test_X = test_df["comment_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(train_X)+list(val_X)+list(test_X))

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



# inp = Input(shape=(maxlen,))

# x = Embedding(max_features, embed_size)(inp)

# x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

# x = GlobalMaxPool1D()(x)

# x = Dense(16, activation="relu")(x)

# x = Dropout(0.1)(x)

# x = Dense(1, activation="sigmoid")(x)

# model = Model(inputs=inp, outputs=x)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# print(model.summary())

# model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))





# define the model

model = Sequential()

model.add(Embedding(max_features, embed_size, input_length=maxlen))



model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

model.add(MaxPooling1D(pool_size=2))



model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score,roc_curve, auc,  f1_score

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import f1_score

# Making the Confusion Matrix

def get_metrics(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)



    class_names=[0,1] # name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')





    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



    # Model Precision: what percentage of positive tuples are labeled as such?

    print("Precision:",metrics.precision_score(y_test, y_pred))



    # Model Recall: what percentage of positive tuples are labelled as such?

    print(" True positive rate or (Recall or Sensitivity) :",metrics.recall_score(y_test, y_pred))



    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn+fp)



    #Specitivity. or True negative rate

    print(" True Negative rate or Specitivity :",specificity)



    false_negative = fn / (fn+tp)



    #False negative rate

    print(" False Negative rate :",false_negative)



    #False positive rate

    print(" False positive rate (Type 1 error) :",1 - specificity)

    

    print('F Score', f1_score(y_test, y_pred))

    print(cm)

# val_pred_y = model.predict_classes([val_X], batch_size=1024, verbose=1)

# get_metrics(val_y,val_pred_y)
fast_text = ['../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec']



fast_text_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in fast_text], axis=-1)
## some config values 

embed_size = 50 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a question to use



model = Sequential()

model.add(Embedding(len(fast_text_matrix), 300, input_length=maxlen, weights=[fast_text_matrix], trainable=False))



model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model.add(GlobalMaxPooling1D())



# model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))



model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

model.fit(train_X, train_y, batch_size=512, epochs=20, validation_data=(val_X, val_y))



get_metrics(val_y,model.predict_classes([val_X], batch_size=1024, verbose=1))



fast_text_val_pred_y = model.predict([test_X], batch_size=1024, verbose=1)

del fast_text_matrix

gc.collect()


glove = ['../input/glove840b300dtxt/glove.840B.300d.txt']

glove_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in glove], axis=-1)

## some config values 

embed_size = 50 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a question to use



model = Sequential()

model.add(Embedding(len(glove_matrix), 300, input_length=maxlen, weights=[glove_matrix], trainable=False))



model.add(SpatialDropout1D(0.2))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))



model.add(GlobalMaxPooling1D())



# model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

model.fit(train_X, train_y, batch_size=512, epochs=20, validation_data=(val_X, val_y))



get_metrics(val_y,model.predict_classes([val_X], batch_size=1024, verbose=1))



glove_val_pred_y = model.predict([test_X], batch_size=1024, verbose=1)

del glove_matrix

gc.collect()
final_predictions =  (fast_text_val_pred_y + glove_val_pred_y) / 2

final_predictions = np.where(final_predictions >= 0.5, 1, 0)

output = pd.DataFrame({"id":test_df["id"].values})

output['prediction'] = final_predictions

output.to_csv("submission.csv", index=False)