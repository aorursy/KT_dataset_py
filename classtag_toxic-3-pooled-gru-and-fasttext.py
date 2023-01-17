import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_union

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import Callback, EarlyStopping
from nltk import word_tokenize
# read data to dataframe
df_train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

print('train shape {} rows, {} cols'.format(*df_train.shape))
print('test shape {} rows, {} cols'.format(*df_test.shape))
X_train = df_train["comment_text"].fillna("fillna").values
y_train = df_train[[
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]].values
X_test = df_test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

EMBEDDING_FILE = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'


def get_coefs(word, *arr):
  return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(
    get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
  if i >= max_features:
    continue
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
from sklearn.metrics import roc_auc_score

class RocAucEvaluation(Callback):

  def __init__(self, validation_data=(), interval=1):
    super(Callback, self).__init__()

    self.interval = interval
    self.X_val, self.y_val = validation_data

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      score = roc_auc_score(self.y_val, y_pred)
      print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))
def get_model():
  inp = Input(shape=(maxlen,))
  x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
  x = SpatialDropout1D(0.2)(x)
  x = Bidirectional(GRU(80, return_sequences=True))(x)
  avg_pool = GlobalAveragePooling1D()(x)
  max_pool = GlobalMaxPooling1D()(x)
  conc = concatenate([avg_pool, max_pool])
  outp = Dense(6, activation="sigmoid")(conc)

  model = Model(inputs=inp, outputs=outp)
  model.compile(
      loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model
model = get_model()
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.summary()
batch_size = 32
epochs = 500

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
roc_auc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, 
                 batch_size=batch_size, 
                 epochs=epochs, 
                 validation_data=(X_val, y_val),
                 callbacks=[roc_auc, EarlyStopping(patience=10)], verbose=2)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
y_pred = model.predict(x_test, batch_size=1024)
submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)