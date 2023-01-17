import pandas as pd
import numpy as np
import re
from tqdm import tqdm_notebook

from nltk.corpus import stopwords

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 300      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "../input/glove840b300dtxt/glove.840B.300d.txt"  
train = pd.read_csv('../input/toxic-classification-trainset/train.csv')
test = pd.read_csv('../input/toxic-classification-testset/test.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
appo = {
"aren't" : "are not",
"can't" : "can not",
"couldn't" : "could not",
"didn't" : "did not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "i would",
"i'd" : "i had",
"i'll" : "i will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't" : "was not",
"we'll" : " will",
"tryin'" : "trying",
"yay!" : " good ",
"yay" : " good ",
"yaay" : " good ",
"yaaay" : " good ",
"yaaaay" : " good ",
"yaaaaay" : " good ",
":/" : " bad ",
":&gt;" : " sad ",
":')" : " sad ",
":-(" : " frown ",
":(" : " frown ",
":s": " frown ",
":-s": " frown ",
"&lt;3": " heart ",
":d": " smile ",
":p": " smile ",
":dd": " smile ",
"8)": " smile ",
":-)": " smile ",
":)": " smile ",
";)": " smile ",
"(-:": " smile ",
"(:": " smile ",
":/": " worry ",
":&gt;": " angry ",
":')": " sad ",
":-(": " sad ",
":(": " sad ",
":s": " sad ",
":-s": " sad ",
r"\br\b": "are",
r"\bu\b": "you",
r"\bhaha\b": "ha",
r"\bhahaha\b": "ha",
r"\bdon't\b": "do not",
r"\bdoesn't\b": "does not",
r"\bdidn't\b": "did not",
r"\bhasn't\b": "has not",
r"\bhaven't\b": "have not",
r"\bhadn't\b": "had not",
r"\bwon't\b": "will not",
r"\bwouldn't\b": "would not",
r"\bcan't\b": "can not",
r"\bcannot\b": "can not",
r"\bi'm\b": "i am",
"m": "am",
"r": "are",
"u": "you",
"haha": "ha",
"hahaha": "ha",
"doesn't": "does not",
"cannot": "can not",
"its" : "it is",
"'s" : " is",
"d'aww!":"cute"
}
keys = [i for i in appo.keys()]

new_train_data = []
ltr = train["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = appo[j]
        xx += j + " "
    new_train_data.append(xx)
    
train["comment_text"] = new_train_data
new_test_data = []
lte = test["comment_text"].tolist()

for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = appo[j]
        xx += j + " "
    new_test_data.append(xx)

test["comment_text"] = new_test_data
trate = train["comment_text"].tolist()
tete = test["comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
    trate[i] = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",trate[i]) #remove ip
    trate[i] = re.sub("\[\[.*\]","",trate[i]) #remove user_name
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
    tete[i] = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",tete[i])
    tete[i] = re.sub("\[\[.*\]","",tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
X_train = list(train["comment_text"])
y_train = train[labels].values
X_test = list(test["comment_text"])

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,lower=True)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))
data = pad_sequences(X_train, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(X_test, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
embeddings_index = {}
f = open(GLOVE_DIR)
print('Loading GloVe from:', GLOVE_DIR,'...', end='')

for line in f:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...")
print(f'Found {len(embeddings_index)} word vectors', end="")

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedding_layer = Embedding(len(word_index) + 1,
#                            EMBEDDING_DIM,
#                            weights = [embedding_matrix],
#                            input_length = MAX_SEQUENCE_LENGTH,
#                            trainable=False,
#                            name = 'embeddings')
# embedded_sequences = embedding_layer(sequence_input)
# x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)
# x = GlobalMaxPool1D()(x)
# x = Dropout(0.1)(x)
# x = Dense(50, activation="relu")(x)
# x = Dropout(0.1)(x)
# preds = Dense(6, activation="sigmoid")(x)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights = [embedding_matrix],
                           input_length = MAX_SEQUENCE_LENGTH,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool]) 
preds = Dense(6, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
tf.keras.utils.plot_model(model)
batch_size = 128
epochs = 4
X_tra, X_val, y_tra, y_val = train_test_split(data, y_train, train_size=0.8, random_state=233)
# filepath="../input/best-model/best.hdf5"
filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val,checkpoint, early]
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
print('Predicting....')
y_pred = model.predict(x_test,batch_size=1024,verbose=1)
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
print('Predicting....')
y_pred = model.predict(data_test,batch_size=1024,verbose=1)
submission = pd.read_csv('../input/toxic-comment/submission_gru.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('preprocess_gru_submission.csv', index=False)
 #plotting Loss
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.plot(hist_adam.history['loss'], color='b', label='Training Loss')
plt.plot(hist_adam.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')