# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

%matplotlib inline
Train = '../input/train/'
Test = '../input/test/'
Labels = '../input/train.csv'
Sample = '../input/sample_submission.csv'
df1 = pd.read_csv(Labels)
df1.Id.value_counts()
train_names = list(f[:36] for f in os.listdir(Train))
test_names = list(f[:36] for f in os.listdir(Test))
print('Training Data Length - {}'.format(len(train_names)))
print('Test Data Length - {}'.format(len(test_names)))
i=34
# We can check what is the Id of the whale and mark it while plotting the Image
title = df1[df1['Image']==train_names[i]]
img = mpimg.imread(os.path.join(Train,train_names[i]))
label =  'Whale Id -' + title['Id'].to_string().split('   ')[1]
plt.title(label)
plt.imshow(img)
print(title)
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        count += 1
    
    return X_train
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
X = prepareImages(df1, df1.shape[0], "train")
X /= 255
y, label_encoder = prepare_labels(df1['Id'])
model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))
model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(Conv2D(64, (5, 5), strides = (1, 1), name = 'conv1'))
model.add(BatchNormalization(axis = 3, name = 'bn1'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv2"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping
# define early stopping callback
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks = [earlystop]
history = model.fit(X, y, epochs=100, batch_size=150, verbose=1,validation_split=0.20,callbacks=callbacks)
col = ['Image']
test_df = pd.DataFrame(test_names, columns=col)
test_df['Id'] = ''
X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255
predictions = model.predict(np.array(X), verbose=1)

for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.to_csv('submission.csv', index=False)
