!pip install transformers==3.0.2
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import gc
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
import tensorflow as tf
import transformers

from sklearn.model_selection import KFold
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
transformers.__version__
from transformers import AutoTokenizer, TFAutoModel
train_df = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
test_df = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")
epochs = 3
maxlen = 50

model_name = "jplu/tf-xlm-roberta-large"

batch_size = 16 * strategy.num_replicas_in_sync
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_df.premise.values
list(train_df.premise.values[:10]), list(train_df.hypothesis.values[:10])
%%time
train_encode = tokenizer(list(train_df.premise.values), list(train_df.hypothesis.values), 
                      max_length=maxlen, return_tensors="np", padding=True, 
                      return_token_type_ids=True, return_attention_mask=True)
def get_model(maxlen=50):
    
    #base_model = TFDistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
    
    base_model = TFAutoModel.from_pretrained(model_name)
    
    input_ids = tf.keras.layers.Input(shape =(maxlen, ), dtype=tf.int32, name="input_ids")
    input_type = tf.keras.layers.Input(shape =(maxlen, ), dtype=tf.int32, name="token_type_ids")
    input_mask = tf.keras.layers.Input(shape =(maxlen, ), dtype=tf.int32, name="attention_mask")
    
    
    embedding = base_model([input_ids, input_mask, input_type])[0]
    #embedding = base_model([input_ids, input_mask])[0]
    
    print(embedding.shape)
    
    output = tf.keras.layers.Dense(3, activation="softmax")(embedding[:, 0, :])
    
    model = tf.keras.models.Model(inputs=[input_ids, input_mask, input_type], outputs = output)
    
    model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
    
    return model
with strategy.scope():
    cls_model = get_model(maxlen)
    cls_model.summary()
%%time
ps = cls_model([train_encode['input_ids'][:10], train_encode['attention_mask'][:10], train_encode['token_type_ids'][:10]])
fold = KFold(n_splits=3, shuffle=True, random_state=108)
%%time
hists = []
models = []
for i, (train_idx, val_idx) in enumerate(fold.split(np.arange(train_df.label.shape[0]))):
    print(f"----FOLD: {i+1}----\n",train_idx, val_idx)
    
    
    x_train = [train_encode['input_ids'][train_idx], 
               train_encode['attention_mask'][train_idx], 
               train_encode['token_type_ids'][train_idx]]
    
    y_train = train_df.label.values[train_idx]
    
    x_val = [train_encode['input_ids'][val_idx],
             train_encode['attention_mask'][val_idx],
             train_encode['token_type_ids'][val_idx]]
    y_val = train_df.label.values[val_idx]
    
    
    hist=cls_model.fit(x_train, y_train,
                       epochs=epochs, 
                       batch_size = batch_size,
                       validation_data = (x_val, y_val),
                      )
    hists.append(hist)
    #models.append(cls_model)
    
    gc.collect()
gc.collect()
%%time
test_encode = tokenizer(list(test_df.premise.values), list(test_df.hypothesis.values), 
                      max_length=maxlen, return_tensors="tf", padding=True, 
                      return_token_type_ids=True, return_attention_mask=True)
"""
preds = []
for model in models:
    ps = model.predict([test_encode['input_ids'], test_encode['attention_mask'], test_encode['token_type_ids']],
                      verbose=1, batch_size=batch_size)
    preds.append(ps)
"""
#ps = np.mean(np.stack(preds), 0)
ps = cls_model.predict([test_encode['input_ids'], test_encode['attention_mask'], test_encode['token_type_ids']],
                      verbose=1, batch_size=batch_size)
submission = test_df.id.copy().to_frame()
submission['prediction'] = np.argmax(ps, 1)
submission.head()
submission.to_csv("submission.csv", index = False)
