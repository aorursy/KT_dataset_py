from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Dense, LSTM, \
Bidirectional, Lambda, Conv1D, MaxPooling1D, GRU,GlobalMaxPooling1D,GlobalAveragePooling1D, concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
class CyclicLR(Callback):
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
data = pd.read_csv("../input/nlp-getting-started/train.csv")
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(data['text'].values)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

pad_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
embeddings_index = {}
f = open('../input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()
print('\nFound %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Found %s word vectors.' % len(embedding_matrix))
del embedding_vector,embeddings_index
X_train,X_test, y_train, y_test = train_test_split(pad_text,data['target'].values,
                                                   test_size=0.33,shuffle=True,random_state=124, stratify=data['target'])
input_text = Input(shape=(60,),dtype='int64')

embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False, mask_zero=True)(input_text)
text_embed = SpatialDropout1D(0.4)(embedding_layer)

hidden_states = Bidirectional(LSTM(units=300, return_sequences=True))(text_embed)
global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
sentence_embed = global_max_pooling(hidden_states)

dense_layer = Dense(256, activation='relu')(sentence_embed)
output = Dense(1, activation='sigmoid')(dense_layer)

BiLSTM = Model(input_text, output)
BiLSTM.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(6e-6))

lr_callback = CyclicLR()
history = BiLSTM.fit(X_train,y_train, batch_size=5, epochs=3,
                       validation_data=(X_test,y_test), callbacks=[lr_callback])
val_preds = BiLSTM.predict(X_test)
val_preds = np.round(val_preds).astype(int)
print(classification_report(y_test,val_preds,target_names = ['Not Relevant', 'Relevant']))
input_text = Input(shape=(60,),dtype='int64')

embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False, mask_zero=True)(input_text)
text_embed = SpatialDropout1D(0.4)(embedding_layer)
conv_layer = Conv1D(300, kernel_size=3, padding="valid", activation='relu')(text_embed)
conv_max_pool = MaxPooling1D(pool_size=2)(conv_layer)

gru_layer = Bidirectional(GRU(300, return_sequences=True))(conv_max_pool)
sentence_embed = GlobalMaxPooling1D()(gru_layer)

dense_layer = Dense(256, activation='relu')(sentence_embed)
output = Dense(1, activation='sigmoid')(dense_layer)

CNNRNN = Model(input_text, output)
CNNRNN.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(6e-6))

lr_callback = CyclicLR()
history = CNNRNN.fit(X_train,y_train, batch_size=5, epochs=3,
                       validation_data=(X_test,y_test), callbacks=[lr_callback])
val_preds = CNNRNN.predict(X_test)
val_preds = np.round(val_preds).astype(int)
print(classification_report(y_test,val_preds,target_names = ['Not Relevant', 'Relevant']))
input_text = Input(shape=(60,),dtype='int64')

embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False, mask_zero=True)(input_text)
text_embed = SpatialDropout1D(0.4)(embedding_layer)
gru_layer = Bidirectional(GRU(300, return_sequences=True))(text_embed)

conv_layer = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(gru_layer)

avg_pool = GlobalAveragePooling1D()(conv_layer)
max_pool = GlobalMaxPooling1D()(conv_layer)
sentence_embed = concatenate([avg_pool, max_pool])

dense_layer = Dense(256, activation='relu')(sentence_embed)
output = Dense(1, activation='sigmoid')(dense_layer)

RNNCNN = Model(input_text, output)

RNNCNN.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(6e-6))

lr_callback = CyclicLR()
history = RNNCNN.fit(X_train,y_train, batch_size=5, epochs=3,
                       validation_data=(X_test,y_test), callbacks=[lr_callback])
val_preds = RNNCNN.predict(X_test)
val_preds = np.round(val_preds).astype(int)
print(classification_report(y_test,val_preds,target_names = ['Not Relevant', 'Relevant']))

train,_, y_train, _ = train_test_split(data['text'].values,data['target'].values,
                                                   test_size=0.2,shuffle=True,random_state=124, stratify=data['target'])
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(train)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

pad_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
bilstm=BiLSTM.predict(pad_text)
cr = CNNRNN.predict(pad_text)
rc = RNNCNN.predict(pad_text)
prediction = pd.DataFrame({"BiLSTM":bilstm.flatten(),"CR":cr.flatten(),"RC":rc.flatten(),"target":y_train})
## Yeah this network is simple as f***, but still scores 80%.....
clf = Sequential([
    Dense(3,activation = 'relu'),
    Dense(1,activation= 'sigmoid')
])
clf.compile(loss = 'binary_crossentropy',optimizer = Adam(3e-5),metrics = ['acc'])
history =clf.fit(prediction.loc[:,['BiLSTM','CR','RC']].values,prediction.iloc[:,-1],validation_split= 0.1,batch_size = 5,epochs = 32)
test = pd.read_csv("../input/nlp-getting-started/test.csv")
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(data['text'].values)
sequences = tokenizer.texts_to_sequences(test.text.values)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

pad_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
bilstm_test = BiLSTM.predict(pad_text).flatten()
cr_test = CNNRNN.predict(pad_text).flatten()
rc_test = RNNCNN.predict(pad_text).flatten()
test_predictions = clf.predict(np.stack([bilstm_test,cr_test,rc_test],axis=-1)).flatten()
test_predictions = np.round(test_predictions).astype(int)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
submission['target'] = test_predictions
submission.to_csv("/kaggle/working/submission.csv",index = False)