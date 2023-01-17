import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
data = pd.read_csv("/kaggle/input/emotion-classifier-spotle-aithon-dataset/aithon2020_level2_traning.csv")
ip= data[data.columns[1:]]/255
op = data[data.columns[0]]
encoder = LabelEncoder()
encoder.fit(op)
encoded_Y = encoder.transform(op)
dummy_y = np_utils.to_categorical(encoded_Y)
model = Sequential()
af = 'sigmoid'
model.add(Dense(10, activation = af, input_shape=(ip.shape[1],)))    
model.add(Dense(3, activation ='softmax'))    
model.summary()
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])
model.fit(ip, dummy_y, validation_split=0.25, epochs=1000, batch_size=100, verbose=1)
predictions = model.predict(ip)
test_loss, test_acc = model.evaluate(ip, dummy_y)
print('loss =',test_loss,'accuracy =',test_acc)
y = [np.argmax(predictions[i]) for i in range(len(predictions))]
y = encoder.inverse_transform(y)
print(y)
print(ip.shape)
