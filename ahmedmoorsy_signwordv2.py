

#dataset_path = "/Users/ahmedmoorsy/Desktop/sign/dataset/output"

dataset_path1 = "/kaggle/input/sign-17-words"



from glob import glob

import tensorflow as tf

import numpy as np

from collections import Counter

import matplotlib.pylab as plt

import random

random.seed(42)

hand_files1 = glob(dataset_path1 + "/*/*.txt")

#hand_files2 = glob(dataset_path + "/*/*_hand.txt")

print(len(hand_files1))



hand_files = []

hand_files = hand_files1

print(len(hand_files))



frames=[] #list to save frame numbers in txt files

for path in hand_files:

    with open(path, mode = 'r') as t:

        numbers = np.array([float(num) for num in t.read().split()])

        if len(numbers) < 84 and sum(numbers) == 0: continue

        frames.append(int(len(numbers)/84))



count = Counter(frames)



plt.bar(count.keys(), count.values())

plt.xlabel('frame number')

plt.ylabel('count')

plt.title('Histogram')

plt.grid(True)

plt.show()
max_frames_length = 300 * 84

max_frame = 50

X = []

Y = []

for path in hand_files:

    label_name = path.split('/')[-2]

    with open(path, mode = 'r') as t:

        numbers = [float(num) for num in t.read().split()]

        if len(numbers) < 84 and sum(numbers) == 0: continue

        X.append(numbers)

        Y.append(label_name)

    
len(Y), len(X)
pad_features = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post', dtype='float32')

X = []

_Y = []

for number,label in zip(pad_features, Y):

    row=0

    landmark_frame = [] 

    labels = []

    for i in range(0, max_frame):

        landmark_frame.append(number[row:row+84])

        labels.append(label)

        row += 84

    X.append(np.array(landmark_frame))

    _Y.append(np.array(labels))

Y = _Y
X=np.array(X)

Y=np.array(Y)

print(X.shape, Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)

print(len(X_train), len(y_train), len(y_test), len(X_test))

y_train = np.array(y_train)

y_test = np.array(y_test)

X_train = np.array(X_train)

X_test = np.array(X_test)



print(X_train.shape)
y_train = y_train.flatten() 

y_test = y_test.flatten() 

y_train = list(y_train)

y_test = list(y_test)

k = set(y_test)

ks = sorted(k)

text=""

for i,word in enumerate(ks):

    if i == len(ks) -1:

        text += word

    else:

        text += word + " "



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import to_categorical

s = Tokenizer()

s.fit_on_texts([text])



encoded=s.texts_to_sequences([y_test])[0]

encoded1=s.texts_to_sequences([y_train])[0]

y_test = to_categorical(encoded)

y_train = to_categorical(encoded1)

y_test = np.array(y_test)

y_train = np.array(y_train)

print(y_train.shape[1])
y_train = np.reshape(y_train, (628, -1, y_train.shape[1])) 

y_test = np.reshape(y_test, (210, -1, y_test.shape[1]))

print(y_train.shape)
from tensorflow.keras import layers, models

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf

from tensorflow.keras.optimizers import Adam

def build_model(label):

    model = Sequential()

    model.add(layers.Masking(input_shape=(max_frame, 84)))

    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True

                  )))

    model.add(layers.Dropout(0.5))

    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True

                  )))

    model.add(layers.Dropout(0.5))

    model.add(layers.TimeDistributed(layers.Dense(label, activation='softmax')))

    optimizer = Adam(0.0001)

    

    model.compile(loss='categorical_crossentropy',

                  optimizer=optimizer,

                  metrics=['accuracy'])

    return model



model=build_model(y_train.shape[2])

print(model.summary())

print('Training stage')

print('==============')

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)

checkpointer = tf.keras.callbacks.ModelCheckpoint('ModelCheckPoint.h5',verbose=1,save_best_only=True)



history=model.fit(X_train, y_train, epochs=100, batch_size=16,validation_data=(X_test,y_test),verbose=1,callbacks=[early_stopping, checkpointer])

score, acc = model.evaluate(X_test, y_test,batch_size=16,verbose=0)

print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
model.save('model.h5')
model = tf.keras.models.load_model('ModelCheckPoint.h5')

score, acc = model.evaluate(X_test, y_test,batch_size=16,verbose=0)

print('Test performance: accuracy={0}, loss={1}'.format(acc, score))
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_converter.experimental_new_converter = True

tflite_model = tflite_converter.convert()

open("tf_lite_model.tflite", "wb").write(tflite_model)


InputSize = 15

MaxLen = 64

HiddenSize = 16



OutputSize = 8

n_samples = 1000



model1 = Sequential()

model1.add(layers.LSTM(HiddenSize, return_sequences=True, input_shape=(MaxLen, InputSize)))

model1.add(layers.TimeDistributed(layers.Dense(OutputSize, activation='softmax')))

model1.compile(loss='categorical_crossentropy', optimizer='adam')





X = np.random.random([n_samples,MaxLen,InputSize])

Y1 = np.random.random([n_samples,OutputSize])

Y1 = np.expand_dims(Y1,1)

print(X.shape, Y1.shape)

model1.fit(X, Y1, batch_size=128, nb_epoch=1)



print(model1.summary())
