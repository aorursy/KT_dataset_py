import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install tensorflow-gpu==1.15.0
import tensorflow as tf
tf.__version__
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import tensorflow_hub as hub
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
text = ['word1','word2','word3']
X = 10*text
NewX = []
for i in range(32000):
    NewX.append(X)
y = np.round(np.random.randn(32000,30))
y = y.reshape(y.shape[0],y.shape[1],1)
batch_size = 32
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Lambda, Input
def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[30])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]
input_text = Input(shape=(30,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(30, 1024))(input_text)

# Complete the model architecture here:

x = LSTM(units=128, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2)(embedding)
x_add_d = Dropout(0.3)(x)
out_1 = TimeDistributed(Dense(1, activation="sigmoid"))(x_add_d)


model2 = Model(inputs=[input_text], outputs=[out_1])
model2.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['accuracy'])
batch_size = 32
model2.fit(np.array(NewX), y, batch_size=batch_size, epochs=5, verbose=1)
