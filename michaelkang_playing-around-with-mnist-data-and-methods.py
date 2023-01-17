# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy.ndimage.filters import gaussian_filter



from sklearn.decomposition import PCA

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score



from sklearn.linear_model import SGDClassifier



from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
#Data is of 28x28 greyscale image. train has 1 more feature 'label' to indicate number

#Other features are labeled pixel0, pixel1, ...,  pixelX

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Now making features and label sets

X_train_df = train.drop('label', axis=1)

Y_train_df = train['label']

X_test_df = test

X_train_df.shape, Y_train_df.shape, X_test_df.shape
#Getting a look at the data

i=50

img=X_train_df.iloc[i].as_matrix()

img=img.reshape((28,28))

img=gaussian_filter(img, sigma=1)               #I'm not sure if this helps that much...

plt.imshow(img,cmap='gray')

plt.title(X_train_df.iloc[i,0])

#This is SGD with just unprocessed data. Note that SVM and Kneighbors takes way too long to run and therefore isn't included on notebook. implementatin of algorithm is the same as below.

sgd = SGDClassifier()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results = cross_val_score(sgd, X_train_df, Y_train_df, cv=cv)

print("SGD Results: %.2f%% +/-%.2f%%" % (results.mean()*100, results.std()*100))
#Manually splitting training set into cross-validation sets using train_test_split

X_train_df_val, X_test_df_val, Y_train_df_val, Y_test_df_val = train_test_split(X_train_df, Y_train_df, train_size = 0.75, random_state = 46)

X_train_df_val.shape, X_test_df_val.shape, Y_train_df_val.shape, Y_test_df_val.shape
#checking to see if using accuracy_score and train_test_split will have a significant effect.

sgd = SGDClassifier()                                           #SGD Results: 89.20%

sgd.fit(X_train_df_val, Y_train_df_val)                         #No real significant difference.

Y_pred = sgd.predict(X_test_df_val)                                 #The art of "ehhhh close enough"

accuracy = round(accuracy_score(Y_test_df_val, Y_pred)*100, 2)

accuracy
#Trying out the simplification that I saw on some other kernels

X_train_df_binary = X_train_df

X_test_df_binary = X_test_df

X_train_df_binary[X_train_df_binary>0]=1

X_test_df_binary[X_test_df_binary>0]=1
#Now fitting the binarized data into SGD

sgd = SGDClassifier()

cv = KFold(n_splits=5,shuffle=True,random_state=42)

results = cross_val_score(sgd, X_train_df_binary, Y_train_df, cv=cv)

print('SGD Results: %.2f%% +/-%.2f%%' % (results.mean()*100, results.std()*100))
#Now working with neural networks. To have reproducibility, we set randomizer seed

seed = 7

np.random.seed(seed)



#Keras requires inputs to be numpy arrays, not pandas dataframes.

temp_data = train.values



X_train = temp_data[:,1:].astype(float)

Y_train = temp_data[:,0]

X_test = test.values

#Noapplying one hot encoding to numpy array Y_train

#using keras packaging since it seems to work better without indexing errors...

encoder = LabelEncoder()

encoder.fit(Y_train)

dummy_y = np_utils.to_categorical(Y_train)
model = Sequential()              #Neural Network Results: 96.66% +/-0.26%

model.add(Dense(200, input_dim=784, kernel_initializer='normal', activation='relu'))

model.add(Dense(30, kernel_initializer='normal', activation='relu'))

model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    



#Checkpointing

checkpoint = ModelCheckpoint('mnist.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]



#Plot learning curves

model_history = model.fit(X_train, dummy_y, validation_split=0.33, epochs=10, batch_size=10, verbose=0)

plt.plot(model_history.history['acc'])

plt.plot(model_history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')
#Now using a Convoluted Neural Netwok with feeds being the images themselves

X_train_2D = X_train.reshape(X_train.shape[0], 28, 28, 1)      #Thus the images have dimensions 1x28x28 (depth 1 since no color)

#plt.imshow(X_train_2D[3],cmap='gray').     #The image was reshaped correctly

X_test_2D = X_test.reshape(X_test.shape[0], 28, 28, 1)

#Now normalize each value between 0 and 1 by dividing by 255.

    #Will have to learn more about #standardScaler and pipeline, since I didn't get high accuracy when used on the above neural network

X_train_2D = X_train_2D / 255

X_test_2D = X_test_2D / 255
model = Sequential()                                                            #These dimensions are obtained using print(model.output_shape)

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28, 1)))    #(None, 26, 26, 32)

model.add(Convolution2D(32, 3, 3, activation='relu'))                           #(None, 24, 24, 32)

model.add(MaxPooling2D(pool_size=(2,2)))                                        #(None, 12, 12, 32)

model.add(Dropout(0.25))



model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

    #Compiling now

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#Checkpointing

checkpoint = ModelCheckpoint('mnist_convolutional.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]



#Plot learning curves

conv_model_history = model.fit(X_train_2D, dummy_y, batch_size = 10, epochs=5, validation_split=0.33, verbose=2, callbacks=callbacks_list)

plt.plot(conv_model_history.history['acc'])

plt.plot(conv_model_history.history['val_acc'])

plt.title('convolutional model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

#Now trying PCA

pca = PCA(whiten=True)

pca.fit(X_train)

variance = pd.DataFrame(pca.explained_variance_ratio_)

np.cumsum(pca.explained_variance_ratio_)

#We see that we have around 674...

pca = PCA(n_components=674,whiten=True)

pca = pca.fit(X_train_df)

X_train_PCA = pca.transform(X_train_df)