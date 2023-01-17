import numpy as np

from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt



from keras.layers import Dense

from keras.models import Sequential

from keras import utils



from sklearn.model_selection import train_test_split
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
X = train_data.drop('label', axis=1)

y = train_data['label']
X = X.values.reshape(X.shape[0],-1)

y = utils.to_categorical(y, 10)
test_data = test_data.values.reshape(test_data.shape[0],-1)

test_data.shape
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=4)
model = Sequential()

model.add(Dense(512, activation='sigmoid', input_shape=(784,)))

model.add(Dense(512, activation= 'sigmoid'))

model.add(Dense(256, activation= 'sigmoid'))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50)
rows = 2

cols = 5

plt.figure(figsize=(10,10))

for x in range(rows * cols):

    pred = np.argmax(model.predict(x_val[x].reshape(1,784)))

    plt.subplot(rows, cols, x+1)

    plt.imshow(x_val[x].reshape(28,28))

    plt.xlabel('Predicted: {}'.format(pred))

plt.tight_layout()

plt.show()
y_pred = model.predict(test_data)
predictions = []

for i,pred in enumerate(y_pred):

    predictions.append(np.argmax(pred))
submission['Label'] = predictions
submission.to_csv('mnist.csv', index= False)