%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, GRU, GlobalMaxPooling1D, Dense
DATA_DIR = '../input/jigsaw-toxic-comment-classification-challenge/'
FASTTEXT_FILE_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
!ls {DATA_DIR}
train_df = pd.read_csv(f'{DATA_DIR}train.csv')
train_df.head()
test_df = pd.read_csv(f'{DATA_DIR}test.csv')
test_df.head()
train_df.shape, test_df.shape
train_word_count = train_df['comment_text'].str.split().apply(lambda x: len(x))
test_word_count = test_df['comment_text'].str.split().apply(lambda x: len(x))
train_word_count.hist(bins=10, rwidth=0.9)
test_word_count.hist(bins=10, rwidth=0.9)
maxlen = 175
print(train_word_count[train_word_count < maxlen].count()/train_word_count.count())
print(test_word_count[test_word_count < maxlen].count()/test_word_count.count())
train_df.isnull().sum()
test_df.isnull().sum()
max_features = 100000
train_text = train_df['comment_text']

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_text.values)

train_sequences = tokenizer.texts_to_sequences(train_text)
train_data = pad_sequences(train_sequences, maxlen=maxlen)
label_names = train_df.columns[2:].values; label_names
target = train_df[label_names]
target.shape
val_count = 15000

x_val = train_data[:val_count]
x_train = train_data[val_count:]
y_val = target[:val_count]
y_train = target[val_count:]
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=1024, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
roc_auc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
class WordEmbeddingsProcessor:
    
    def __init__(self, file_path, max_features, emb_sz, toknzr):
        self.file_path = file_path
        self.max_features = max_features
        self.emb_sz = emb_sz
        self.toknzr = toknzr
        self.embeddings_index = {}
        
    def generate_embeddings_index(self):
        with open(self.file_path, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                
    def get_embedding_matrix(self):
        self.generate_embeddings_index()
        
        word_index = self.toknzr.word_index
        num_words = min(self.max_features, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.emb_sz))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        return embedding_matrix
emb_sz = 300

fastTextProcessor = WordEmbeddingsProcessor(FASTTEXT_FILE_PATH,
                                           max_features=max_features,
                                           emb_sz=emb_sz,
                                           toknzr=tokenizer)
emb_matrix = fastTextProcessor.get_embedding_matrix()
def build_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, emb_sz, weights=[emb_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(128, dropout=0.3, recurrent_dropout=0.5,  return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=x)
    return model


model = build_model()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
best_weights_path = 'weights_base.best.hdf5'
val_loss_checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=4)
# takes too long to run in kernel environment, so I commented out this section.
# Instead I will load pretrained weights.

# model.fit(x_train, y_train,
#          epochs=20,
#          batch_size=1024,
#          validation_data=(x_val, y_val),
#          callbacks=[roc_auc, val_loss_checkpoint, early_stop], verbose=1)
# loading pretrained weights
# make sure you comment out this section when you train the model
# After running 19 epochs, I got validation loss of 0.03867 and ROC-AUC 0.989902 on validation set

pretrained_weights_path = '../input/toxic-pretrained-gru-weights/pretrained.best.hdf5'
model.load_weights(pretrained_weights_path)
val_preds = model.predict(x_val, batch_size=1024, verbose=1)
roc_auc_score(y_val, val_preds)
test_sequences = tokenizer.texts_to_sequences(test_df['comment_text'])
x_test = pad_sequences(test_sequences, maxlen=maxlen)
# uncomment to make test set predictions
#test_preds = model.predict(x_test, batch_size=1024, verbose=1)
# once you make test set predictions, uncomment this section to create submission file

# sub_df = pd.DataFrame(test_preds, columns=label_names)
# sub_df.insert(0, 'id', test_df['id'])
# sub_df.head()