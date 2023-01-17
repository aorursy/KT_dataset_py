import numpy as np

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import operator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import SimpleRNN

from tensorflow.keras.layers import RepeatVector

from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model
valid_characters = '0123456789.+*-/'
val_char_dict = dict((character, index) for index, character in enumerate(valid_characters))
val_char_dict_inv = dict((index, character) for index, character in enumerate(valid_characters))
def one_hot_encode(decoded):

  encoded = np.zeros((repeat_steps, len(valid_characters)))

  padding = repeat_steps - len(decoded)



  for index, character in enumerate(decoded):

    encoded[index+padding, val_char_dict[character]] = 1

  

  for index in range(0, padding):

    encoded[index, val_char_dict['0']] = 1

  

  return encoded
def one_hot_decode(encoded):

  decoded = [val_char_dict_inv[np.argmax(array)] for index, array in enumerate(encoded)]

  decoded = ''.join(decoded)

  

  return decoded
number_max = 100 #Up to this number

repeat_steps = len(str(number_max-1)) * 2 + 1
operators = ['+', '*', '-', '/']

operators_dict = { "+":operator.add, 

                  '*':operator.mul, 

                  "-":operator.sub,

                  '/':operator.truediv}
def data_generator():

  number_1 = np.random.randint(1,number_max)

  operator_index = np.random.randint(0,len(operators))

  operator = operators[operator_index]

  number_2 = np.random.randint(1,number_max)

  operation = str(number_1) + operator + str(number_2)

  result = str(round(operators_dict[operator](number_1,number_2),5))

  return operation, result
data_points = 1000000

test_size = 0.2



training_size = int(round(data_points * (1-test_size),0))

test_size = data_points - training_size



x_train = []

x_test = []

y_train = []

y_test = []



for i in tqdm(range(0, training_size)):

  x, y = data_generator()

  x_e = one_hot_encode(x)

  y_e = one_hot_encode(y)

  x_train.append(x_e)

  y_train.append(y_e)



for i in tqdm(range(0, test_size)):

  x, y = data_generator()

  x_e = one_hot_encode(x)

  y_e = one_hot_encode(y)

  x_test.append(x_e)

  y_test.append(y_e)



x_train = np.array(x_train)

y_train = np.array(y_train)

x_test = np.array(x_test)

y_test = np.array(y_test)
model = Sequential()



model.add(SimpleRNN(units=1024, input_shape=(None, len(valid_characters))))

model.add(RepeatVector(repeat_steps))



model.add(SimpleRNN(units=1024, return_sequences=True))

model.add(TimeDistributed(Dense(units=len(valid_characters), activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
plot_model(model)
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

save_best = ModelCheckpoint('best_model.hdf5', monitor='val_loss', save_best_only=True, mode='min')
history = model.fit(x_train,

                    y_train, 

                    batch_size=512,

                    epochs=250,

                    validation_data=(x_test, y_test),

                    callbacks=[early_stopping, save_best])
plt.figure(figsize=(10,6))

plt.plot(np.arange(0,len(history.history['loss'])), history.history['loss'], label='train_loss')

plt.plot(np.arange(0,len(history.history['val_loss'])), history.history['val_loss'], label='test_loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.title('Loss vs. Epochs')

plt.legend()

plt.show()
plt.figure(figsize=(10,6))

plt.plot(np.arange(0,len(history.history['accuracy'])), history.history['accuracy'], label='Train Accuracy')

plt.plot(np.arange(0,len(history.history['val_accuracy'])), history.history['val_accuracy'], label='Test Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.title('Accuracy vs. Epochs')

plt.legend()

plt.show()
model = load_model('best_model.hdf5')
def predict_result(operation):

  o = one_hot_encode(operation)

  o = np.reshape(o, (1, o.shape[0], o.shape[1] ))

  predictions = model.predict(o)

  for prediction in predictions:

    return one_hot_decode(prediction).lstrip("0")
predict_result('14+7')
predict_result('99*5')
predict_result('2/8')
predict_result('75-64')