# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.environ["WANDB_API_KEY"] = "0" ## to silence warning
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import tensorflow as tf
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels, frequencies = np.unique(train.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.show()
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_sentence(s):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)
encode_sentence("I love machine learning")
def bert_encode(hypotheses, premises, tokenizer):
    
  num_examples = len(hypotheses)
  
  sentence1 = tf.ragged.constant([
      encode_sentence(s)
      for s in np.array(hypotheses)])
  sentence2 = tf.ragged.constant([
      encode_sentence(s)
       for s in np.array(premises)])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)
max_len = 50

init = tf.keras.initializers.GlorotNormal()
def build_model():
    bert_encoder = TFBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
    
    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, dropout=0.3, return_sequences=True))(embedding)
    #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer=init)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    output = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(lr=3e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
with strategy.scope():
    model = build_model()
    model.summary()
model.fit(train_input, train.label.values, epochs = 8, verbose = 1, batch_size = 64, validation_split = 0.2)
test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)
test_input
predictions = []
for i in range(5195):
    ei = {'input_word_ids': tf.expand_dims(test_input["input_word_ids"][i], 0), 'input_mask':  tf.expand_dims(test_input['input_mask'][i],0), 'input_type_ids':  tf.expand_dims(test_input["input_type_ids"][i],0)}
    predictions.append(np.argmax(model.predict(ei)))
len(predictions)
a = np.array(predictions)
a.shape
submission = test.id.copy().to_frame()
submission['prediction'] = predictions
submission.head()
submission.to_csv("submission.csv", index = False)