import keras
from PIL import Image
Image.open('../input/img.png')
model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=3, input_shape=(2,10)))
model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=3, batch_input_shape=(8,2,10)))
model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=3, input_shape=(2,10), return_sequences=False))

model.summary()
model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=3, batch_input_shape=(8,2,10), return_sequences=False))

model.summary()
model = keras.models.Sequential()

model.add(keras.layers.LSTM(units=3, batch_input_shape=(8,2,10), return_sequences=True))

model.summary()