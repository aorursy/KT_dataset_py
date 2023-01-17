import os
os.listdir('../input')
import pandas as pd
import numpy as np
np.random.seed(42)
# read data to dataframe
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('train shape {} rows, {} cols'.format(*df_train.shape))
print('test shape {} rows, {} cols'.format(*df_test.shape))
df_train.head()
df_test.head()
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in target_cols:
  print(label)
  print(df_train[label].value_counts())
  print('*'*80)
from keras.preprocessing import text, sequence
X_train = df_train["comment_text"].fillna("fillna").values

y_train = df_train[target_cols].values
X_test = df_test["comment_text"].fillna("fillna").values
max_features = 30000
maxlen = 100
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

X_test = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import *


class Position_Embedding(Layer):

  def __init__(self, size=None, mode='sum', **kwargs):
    self.size = size  # 必须为偶数
    self.mode = mode
    super(Position_Embedding, self).__init__(**kwargs)

  def call(self, x):
    if (self.size == None) or (self.mode == 'sum'):
      self.size = int(x.shape[-1])
    batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
    position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
    position_j = K.expand_dims(position_j, 0)
    position_i = K.cumsum(K.ones_like(x[:, :, 0]),
                          1) - 1  # K.arange不支持变长，只好用这种方法生成
    position_i = K.expand_dims(position_i, 2)
    position_ij = K.dot(position_i, position_j)
    position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
    if self.mode == 'sum':
      return position_ij + x
    elif self.mode == 'concat':
      return K.concatenate([position_ij, x], 2)

  def compute_output_shape(self, input_shape):
    if self.mode == 'sum':
      return input_shape
    elif self.mode == 'concat':
      return (input_shape[0], input_shape[1], input_shape[2] + self.size)
class Attention(Layer):

  def __init__(self, nb_head, size_per_head, **kwargs):
    self.nb_head = nb_head
    self.size_per_head = size_per_head
    self.output_dim = nb_head * size_per_head
    super(Attention, self).__init__(**kwargs)

  def build(self, input_shape):
    self.WQ = self.add_weight(
        name='WQ',
        shape=(input_shape[0][-1], self.output_dim),
        initializer='glorot_uniform',
        trainable=True)
    self.WK = self.add_weight(
        name='WK',
        shape=(input_shape[1][-1], self.output_dim),
        initializer='glorot_uniform',
        trainable=True)
    self.WV = self.add_weight(
        name='WV',
        shape=(input_shape[2][-1], self.output_dim),
        initializer='glorot_uniform',
        trainable=True)
    super(Attention, self).build(input_shape)

  def Mask(self, inputs, seq_len, mode='mul'):
    if seq_len == None:
      return inputs
    else:
      mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
      mask = 1 - K.cumsum(mask, 1)
      for _ in range(len(inputs.shape) - 2):
        mask = K.expand_dims(mask, 2)
      if mode == 'mul':
        return inputs * mask
      if mode == 'add':
        return inputs - (1 - mask) * 1e12

  def call(self, x):
    # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
    # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
    if len(x) == 3:
      Q_seq, K_seq, V_seq = x
      Q_len, V_len = None, None
    elif len(x) == 5:
      Q_seq, K_seq, V_seq, Q_len, V_len = x
    # 对Q、K、V做线性变换
    Q_seq = K.dot(Q_seq, self.WQ)
    Q_seq = K.reshape(Q_seq,
                      (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
    Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
    K_seq = K.dot(K_seq, self.WK)
    K_seq = K.reshape(K_seq,
                      (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
    K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
    V_seq = K.dot(V_seq, self.WV)
    V_seq = K.reshape(V_seq,
                      (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
    V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
    # 计算内积，然后mask，然后softmax
    A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head**0.5
    A = K.permute_dimensions(A, (0, 3, 2, 1))
    A = self.Mask(A, V_len, 'add')
    A = K.permute_dimensions(A, (0, 3, 2, 1))
    A = K.softmax(A)
    # 输出并mask
    O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
    O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
    O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
    O_seq = self.Mask(O_seq, Q_len, 'mul')
    return O_seq

  def compute_output_shape(self, input_shape):
    return (input_shape[0][0], input_shape[0][1], self.output_dim)
from keras.callbacks import *
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
S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
embeddings = Position_Embedding()(embeddings)
O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
O_seq = GlobalMaxPooling1D()(O_seq)
O_seq = Dropout(0.8)(O_seq)
outputs = Dense(6, activation='sigmoid')(O_seq)
model = Model(inputs=S_inputs, outputs=outputs)
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from sklearn.model_selection import train_test_split
X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, test_size=0.3, random_state=233)
roc_auc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist = model.fit(
    X_tra,
    y_tra,
    callbacks=[EarlyStopping(patience=10), roc_auc],
    batch_size=32,
    epochs=1000,
    validation_data=(X_val, y_val))
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

submission = pd.read_csv('../input/sample_submission.csv')
submission[target_cols] = y_pred
submission.to_csv('submission.csv', index=False)