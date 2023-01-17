import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv("../input/train.csv")
train_data.shape
test_data = pd.read_csv("../input/test.csv")
test_data.shape
for r in range(10):
    # The first column is the label
    data = train_data.iloc[[r]]
    label = data.iloc[0]['label']
    data.drop(columns="label", inplace=True)



    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(data, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    
test_data.head()
y = train_data["label"]
train_data.drop(columns='label', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2)    
encoder = LabelEncoder()
encoder.fit(y)
y_categorical_train = pd.get_dummies(y_train)
y_categorical_test = pd.get_dummies(y_test)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(250, input_dim=784,activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_categorical_train, epochs=25, verbose=1)
scores = model.evaluate(X_test, y_categorical_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict_classes(test_data)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)
len(prediction_)
submission = pd.read_csv("../input/sample_submission.csv")
submission["label"] = prediction_
submission.to_csv("submit.csv")
