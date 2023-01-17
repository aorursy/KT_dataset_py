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
import tensorflow as tf

from keras.callbacks import EarlyStopping

from transformers import BertTokenizer, TFBertModel

import plotly.subplots as sub

import plotly.graph_objs as go

import seaborn as sns

import matplotlib.pyplot as plt
import os

os.environ["WANDB_API_KEY"] = "0"
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() 

    print('Number of replicas:', strategy.num_replicas_in_sync)
test = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")

train = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
print(train.shape)

train.head()
print("Languages used : {}".format(np.unique(train["language"])))
import plotly.express as px



fig = px.pie(train, values='label', names='language')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
labels = ['entailment', 'contradiction', 'neutral']

values = list(train.label.value_counts())



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
print("For entailment\nPremise : \n{}\nHypothesis : \n{}\n\n".format(train.premise.iloc[0], train.hypothesis.iloc[0]))





print("For neutral\nPremise : \n{}\nHypothesis : \n{}\n\n".format(train[train.language=="English"].premise.iloc[4], train[train.language=="English"].hypothesis.iloc[4]))





print("For contradiction\nPremise : \n{}\nHypothesis : \n{}\n\n".format(train.premise.iloc[1], train.hypothesis.iloc[1]))
temp = train.groupby(["language", "label"]).count()["id"].reset_index()



fig = px.bar(temp, x="language", y="id", color="label", title="Label distribution according to language")

fig.show()
train["input"] = train[["premise", "hypothesis"]].apply(lambda x: " ".join(x), axis=1)

test["input"] = test[["premise", "hypothesis"]].apply(lambda x: " ".join(x), axis=1)



train = train.drop(["premise", "hypothesis"], axis=1)

test = test.drop(["premise", "hypothesis"], axis=1)
import tensorflow as tf

from transformers import TFAutoModel, AutoTokenizer
# model_name = 'bert-base-multilingual-cased'

model_name = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(model_name)
def encode_sentence(s):

    tokens = list(tokenizer.tokenize(s))

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)
def bert_encode(text, tokenizer):

    

    num_examples = len(text)

    text = tf.ragged.constant([

      encode_sentence(s)

       for s in np.array(text)])



    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*text.shape[0]

    input_word_ids = text



    input_mask = tf.ones_like(input_word_ids).to_tensor()



    input_type_ids = tf.zeros_like(text).to_tensor()



    inputs = {

      'input_word_ids': input_word_ids.to_tensor(),

      'input_mask': input_mask,

      'input_type_ids': input_type_ids}



    return inputs
train_input = bert_encode(train.input.values, tokenizer)
max_len = 150



def build_model():

    bert_encoder = TFBertModel.from_pretrained(model_name)

    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2))(embedding)

    output = tf.keras.layers.Dense(3, activation='softmax')(lstm)

    

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    

    return model



with strategy.scope():

    model = build_model()

    model.summary()

early = EarlyStopping(monitor='val_loss',mode='auto', baseline=None, restore_best_weights=False)

model.fit(train_input, train.label.values, epochs = 30, batch_size = 8, validation_split = 0.2, callbacks=[early])

test_input =  bert_encode(test.input.values, tokenizer)

predictions = [np.argmax(i) for i in model.predict(test_input)]



submission = test.id.copy().to_frame()

submission['prediction'] = predictions[:5195]



submission.head()
submission.to_csv('submission.csv',index=False)