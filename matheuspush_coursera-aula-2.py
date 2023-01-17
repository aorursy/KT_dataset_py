import pip

try:
    __import__('keras')
except ImportError:
    pip.main(['install', 'keras']) 
    
try:
    __import__('h5py')
except ImportError:
    pip.main(['install', 'h5py']) 

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

seed = 1337
np.random.seed(seed)
from keras.datasets import reuters

max_words = 1000
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2,
                                                         seed=seed)
num_classes = np.max(y_train) + 1  # 46 topics
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test)
model = Sequential()  # Instantiate sequential model
model.add(Dense(512, input_shape=(max_words,), activation="relu")) # Add first layer. Make sure to specify input shape
model.add(Dropout(0.5)) # Add second layer
model.add(Dense(num_classes, activation="softmax")) # Add third layer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#some learners constantly reported 502 errors in Watson Studio. 
#This is due to the limited resources in the free tier and the heavy resource consumption of Keras.
#This is a workaround to limit resource consumption

from keras import backend as K

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

batch_size = 32
model.fit(x_train, y_train, epochs=5, batch_size=batch_size)
score = model.evaluate(x_test, y_test, batch_size=batch_size)
score[1]
model.save("model.h5")  # upload this file to the grader in the next code block
!base64 model.h5 > model.h5.base64
!rm -f rklib.py
!wget https://raw.githubusercontent.com/IBM/coursera/master/rklib.py
from rklib import submit
key = "XbAMqtjdEeepUgo7OOVwng"
part = "LqPRQ"
email = "matheusferndes@gmail.com"
secret = "XE3LTcxupZeYgkeI"

with open('model.h5.base64', 'r') as myfile:
    data=myfile.read()
submit(email, secret, key, part, [part], data)