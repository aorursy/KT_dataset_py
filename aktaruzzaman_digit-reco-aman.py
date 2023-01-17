import numpy as np
import pandas as pd
import seaborn as sns
from scipy import misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

np.random.seed(3)
import itertools as it
t = pd.read_csv('../input/train.csv')
ts = pd.read_csv('../input/test.csv')
t.isnull().sum().sum()
ts.isnull().sum().sum()
ts = ts/255.0
X = t.drop('label',axis=1)/255.0
Y = t.label
from keras.utils import to_categorical as tc
Y = tc(Y,num_classes=10)
X = X.values.reshape(-1,28,28,1)
ts = ts.values.reshape(-1,28,28,1)
from sklearn.model_selection import train_test_split as tts

X_train,X_test,Y_train,Y_test = tts(X,Y,test_size=0.3,random_state=2)
X_train.shape
plt.imshow(X_train[0][:,:,0])
#import packages
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D,Flatten, LeakyReLU
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='linear', input_shape = (28,28,1)))
model.add(LeakyReLU(alpha=.001))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='linear'))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "linear"))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
from keras import optimizers as op
optimizer = op.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
X_train.shape
# With data augmentation to prevent overfitting (accuracy 0.99286)
from keras.preprocessing.image import ImageDataGenerator as imgen
datagen = imgen(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
X_train.shape
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),
                              epochs = 30, validation_data = (X_test,Y_test),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86
                              , callbacks=[learning_rate_reduction])
# predict results
results = model.predict(ts)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
sub = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
sub.to_csv('Submission.csv',index=False)


