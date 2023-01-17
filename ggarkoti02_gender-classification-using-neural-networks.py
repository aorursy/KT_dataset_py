import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from keras import Sequential

from keras.layers import Dense
data = pd.read_csv('../input/gender-classification-dataset/gender_classification_v7.csv')
data.head()
data.info()
data['gender'] = pd.get_dummies(data['gender'])
X = data.drop('gender', axis=1)

y = data['gender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test  = scaler.transform(X_test)
y_train = y_train.to_numpy()

y_test  = y_test.to_numpy() 
model = Sequential()

model.add(Dense(7, activation='relu', input_dim=7))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=5, validation_data=(X_test, y_test), epochs=10, verbose=1)
def learning_curve(history, epoch):

  epoch_range = range(1, epoch+1)

  plt.plot(epoch_range, history.history['accuracy'])

  plt.plot(epoch_range, history.history['val_accuracy'])

  plt.title('Model Accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('Epochs')

  plt.legend(['Train', 'Val'], loc='upper left')

  plt.show()



  plt.plot(epoch_range, history.history['loss'])

  plt.plot(epoch_range, history.history['val_loss'])

  plt.title('Model Loss')

  plt.ylabel('loss')

  plt.xlabel('Epochs')

  plt.legend(['Train', 'Val'], loc='upper left')

  plt.show()
learning_curve(history, 10)
pred = model.predict(X_test)
for i in range(10):

    print('Actual result: ', y_test[i])

    print('Predicted result: ', pred[i])

    print()