import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        
print(paths)
# Gathering the data
data = pd.read_csv(paths[0])

# The data used for training
chunked_data = []
chunked_labels = []

# Temporaly variable
chunk = []

for i, row in zip(range(1, len(data['BidOpen'])), data.itertuples()):
    chunk.append([ [r] for r in row[3:] ])
    
    if not i % 8:
        chunked_data.append(chunk)
        chunked_labels.append([[[int(chunk[0][0] > chunk[-1][0])]]])
        chunk = []
        
print(f'Data: {len(chunked_data)}\nLabels: {len(chunked_labels)}')
# Normalizing the data
chunked_data = np.array(chunked_data)
chunked_data -= chunked_data.mean()
chunked_data /= chunked_data.std()

chunked_labels = np.array(chunked_labels, dtype=np.float32)

# Shuffling the data
np.random.shuffle(chunked_data)
np.random.shuffle(chunked_labels)

print(f'Data: {chunked_data.shape}\nLabels: {chunked_labels.shape}')
# Machine Learning section

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ConvModel:
    def __init__(self, rl1=32, rl2=32, d1=16, out=1):
        # Create the model
        self.__model = Sequential()
        self.__model.add(Conv2D(rl1, (2, 2), input_shape=[8, 10, 1]))
        self.__model.add(AveragePooling2D((2,2)))
        self.__model.add(Conv2D(rl2, (2, 2)))
        self.__model.add(AveragePooling2D((2,2)))
        self.__model.add(Dropout(0.5))
        self.__model.add(Dense(d1, activation='relu'))
        self.__model.add(Dense(out, activation='softmax'))
        
        # Compiling the model
        self.__model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
    def predict(self, data):
        pred = self.__model.predict(data)
        return pred
    
    def train(self, X, Y):
        history = self.__model.fit(X[:-500], Y[:-500], epochs=15, validation_data=(X[-500:], Y[-500:]), batch_size=2)
cmodel = ConvModel()
cmodel.train(chunked_data, chunked_labels)