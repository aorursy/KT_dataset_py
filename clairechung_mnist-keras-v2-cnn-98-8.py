# Basic data processing

import numpy as np

import pandas as pd 



# Try visualize pixel values as an image

import matplotlib.pyplot as plt



# Split labelled dataset into training and testing data to test & improve our model

from sklearn.model_selection import train_test_split 

# Change label formats between input/human-readable/output-required & better for model training formats

from sklearn.preprocessing import LabelEncoder, LabelBinarizer



# To build our CNN sequential model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D # CNN

from keras import backend as K



# List the data files in store

from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))
# Read in 2 lines of the training data to inspect

!head -n2 ../input/train.csv
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
# Reshape and normalize training data

# drop the "label" column for training dataset

X = train.drop("label",axis=1).values.reshape(-1,1,28,28).astype('float32')/255.0

y = train["label"]



# Reshape and normalize test data

X_test1 = test.values.reshape(-1,1,28,28).astype('float32')/255.0
plt.imshow(train.drop("label",axis=1).iloc[0].values.reshape(28,28),cmap=plt.get_cmap('binary'))

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Before binarization

y_train.head()
lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_test = lb.fit_transform(y_test)
# After binarization

y_train
# Start a Keras sequential model

model = Sequential()

# Before Keras 2: K.set_image_dim_ordering('th')

K.set_image_data_format('channels_first')

# Before Keras 2: model.add(Convolution2D(30,5,5, border_mode= 'valid', input_shape=(1,28,28),activation= 'relu' ))

model.add(Convolution2D(30, (5,5),padding='valid',input_shape=(1,28,28),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(15, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Drop out 20% of training data in each batch

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation= 'relu' ))

model.add(Dense(50, activation= 'relu' ))

model.add(Dense(10, activation= 'softmax' ))

  # Compile model

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
model.fit(X_train, y_train,

          epochs=25,

          batch_size= 128)
score = model.evaluate(X_test, y_test, batch_size=128)
score
y_test1 = model.predict(X_test1)

y_test1
y_test1 = lb.fit_transform(np.round(y_test1))

y_test1
predicted_labels = np.argmax(y_test1, axis=1)

predicted_labels
cmaps = ['binary','gray','summer','YlOrRd_r','Set3','BuGn_r','spring','tab20b','PuRd_r','winter']

fig,axarr = plt.subplots(10,10)



for i,ax in enumerate(axarr.flat):

    ax.imshow(test.iloc[i].values.reshape(28,28),cmap=plt.get_cmap(cmaps[predicted_labels[i]]))

plt.setp(axarr, xticks=[], yticks=[])

plt.subplots_adjust(hspace=-0.2,wspace=-0.2)

plt.show()
np.savetxt('submission_kagglekernel_cnn25epochs.csv', 

           np.c_[range(1,len(X_test1)+1),predicted_labels], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')