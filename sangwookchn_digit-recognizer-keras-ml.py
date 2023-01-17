import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
X_train = (train.iloc[:,1:].values).astype('float32')
Y_train = (train.iloc[:,0].values).astype("float32")
X_test = (test.values.astype("float32"))
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape)
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train)

np.random.seed(10)
for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping 
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
print(Y_train.shape)
model = Sequential()
model.add(Convolution2D(64, 5, 5, input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Convolution2D(128, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 50, activation = 'relu'))
model.add(Dense(output_dim = 10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs =20, batch_size= 128)
predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DigitRecognitions3.csv", index=False, header=True)