# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install tqdm  >> /dev/null
!pip install bert-for-tf2 >> /dev/null
!pip install sentencepiece >> /dev/null
!pip install seaborn
import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

from sklearn.metrics import confusion_matrix, classification_report, f1_score

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
train = pd.read_csv('/kaggle/input/train.csv')
test = pd.read_csv('/kaggle/input/test.csv')
df_gameoverview = pd.read_csv('/kaggle/input/game_overview.csv')
sample_submission = pd.read_csv("/kaggle/input/sample_submission_wgBqZCk.csv")
test['user_suggestion'] = 0
chart = sns.countplot(train.user_suggestion, palette=HAPPY_COLORS_PALETTE)
plt.title("Number of test per user suggestion")
chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right');
!pip install wget
import wget as wget
#url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
url = 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip'
wget.download(url)
#!unzip uncased_L-24_H-1024_A-16.zip
!unzip cased_L-24_H-1024_A-16.zip
os.makedirs("model", exist_ok=True)
#!mv uncased_L-24_H-1024_A-16/ model
!mv cased_L-24_H-1024_A-16/ model

#bert_model_name="uncased_L-24_H-1024_A-16"
bert_model_name="cased_L-24_H-1024_A-16"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
class IntentDetectionData:
  DATA_COLUMN = "user_review"
  LABEL_COLUMN = "user_suggestion"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)
def create_model(max_seq_len, bert_ckpt_file):

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.2)(cls_out)
    logits = keras.layers.Dense(units=1024, activation="relu")(cls_out)
    logits = keras.layers.Dropout(0.2)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)
        
    return model
def create_model(max_seq_len, bert_ckpt_file):

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)
    bert_output_ = tf.keras.layers.Reshape((max_seq_len, 1024))(bert_output)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    #print(cls_out)
    #cls_out = keras.layers.Dropout(0.5)(cls_out)
    #gru_out = tf.keras.layers.LSTM(100, activation='tanh')(bert_output_)
    #print(gru_out.shape)
    logits = keras.layers.Dense(units=768, activation="tanh")(gru_out)
    logits = keras.layers.Dropout(0.5)(logits)
    #logits = keras.layers.Dense(units=256, activation="tanh")(logits)
    #logits = keras.layers.Dropout(0.5)(logits)
    print(logits.shape)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)
    print(logits.shape)
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    print(model)
    load_stock_weights(bert, bert_ckpt_file)
        
    return model
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
classes = train.user_suggestion.unique().tolist()
data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=200)
file_path = "best_model.hdf5"

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
BATCH_SIZE = 8 * tpu_strategy.num_replicas_in_sync
with tpu_strategy.scope():
    model = create_model(data.max_seq_len, bert_ckpt_file)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1)
    print(model.summary())
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)    
    # train model normally
    history = model.fit(
      x=data.train_x, 
      y=data.train_y,
      validation_split=0.1,
      batch_size=BATCH_SIZE,
      shuffle=True,
      epochs=10,
      callbacks=[es]
    )


#model.save('model_bert.h5')
#model.load_weights('best_model.hdf5')
import os
#os.remove("best_model.hdf5")
#os.remove("/kaggle/working/model_bert.h5")
os.remove("/kaggle/working/cased_L-24_H-1024_A-16.zip")
#shutil.rmtree("./data/test")
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Loss over training epochs')
plt.show();
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(history.history['acc'])
ax.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.title('Accuracy over training epochs')
plt.show();
y_pred_test = model.predict(data.test_x).argmax(axis=-1)
sub_file = pd.read_csv('/kaggle/input/sample_submission_wgBqZCk.csv',sep=',')
#End
#make the predictions with trained model and submit the predictions.

print("Prediction Done, let us now create a submission file and write it to csv")
sub_file.user_suggestion=y_pred_test
sub_file['user_suggestion'] = sub_file['user_suggestion'].map({0: 'zero', 1: 'one'})
sub_file['user_suggestion'] = sub_file['user_suggestion'].map({'zero':1, 'one': 0})
sub_file.to_csv('Submission_bert.csv',index=False)
