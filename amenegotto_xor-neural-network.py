import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# inputs
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# outputs
target_data = np.array([[0],[1],[1],[0]], "float32")
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

history=model.fit(training_data, target_data, nb_epoch=500, verbose=2)

print(model.predict(training_data).round())
import matplotlib.pyplot as plt

# list all data in history
#print(history.history.keys())
#print(history.history)

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
