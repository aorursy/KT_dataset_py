import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_data=pd.read_csv('../input/digit-recognizer/train.csv')

train_data.info()

'''
The training data is in the form of a csv file, 
where each digital image is represented by 784 columns. We simply need 
to modify training data of csv file to a 28x28 matrix form 
which will enable CNN to operate on it.
'''

from keras.utils import to_categorical

y_train = train_data['label'] # we need to predict labels 
y_train = to_categorical(y_train.values, num_classes=10) # 0-9 prediction hence 10 classes
print(y_train.shape)

X_train = train_data.drop(labels = ['label'], axis = 1) #other than labels (including pixels only) 
X_train = X_train.to_numpy()  # to numpy array (CNN requires numpy array not dataframes)

# normalise values into [0,1] range to speed up training
X_train = X_train/255

# resize to represent 28x28 image
# x_train shape changes from (42000,784) to (42000, 28, 28, 1) 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(X_train.shape)

# printing images from training data
for i in range(8):
    plt.subplot(420 + 1 + i)
    plt.imshow(X_train[i][:,:,0])
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot
cnn = Sequential()
cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu', input_shape=(28,28,1))) # convolution layer with relu activation
cnn.add(MaxPool2D()) #pooling
cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu')) # second convolution layer
cnn.add(MaxPool2D()) # pooling
cnn.add(Flatten()) # flattening
cnn.add(Dense(units = 120, activation = 'relu')) # full connection
cnn.add(Dense(units = 10, activation = 'softmax')) # output layer 

cnn.compile('adam', 'categorical_crossentropy',metrics = ['acc']) # categorical since different classes for images 
cnn.summary()
# Early stopping dependent on validation accuracy
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)

model = cnn.fit(X_train, y_train, validation_split = 0.2, epochs = 500, callbacks = [es])
plt.plot(model.history['acc'], label='train_data')
plt.plot(model.history['val_acc'], label='val')
plt.legend()
plt.show()
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
test_data = test_data.to_numpy()
test_data = test_data/255
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)
pred = cnn.predict(test_data)
pred = np.argmax(pred,axis = 1)
pred
'''
To submit on kaggle

'''


tid = np.arange(len(pred)) + 1
label= pd.Series(pred)
submit = pd.DataFrame({ 'ImageId' : tid, 'Label': label})
submit.head()
submit.to_csv('digit_recognizer_cnn3.csv', index = False)
