# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from skimage.io import imread,imshow

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  



# Any results you write to the current directory are saved as output.
test= pd.read_csv('../input/firstaa/submission.csv')
test['has_cactus']=test['has_cactus'].apply(lambda x:1 if x>=0.3 else 0)
# train = pd.read_csv('../input/train.csv')

# test = pd.read_csv('../input/sample_submission.csv')

# y_train = np.array(train.has_cactus)
len(train)
from skimage.io import imread

y_train = np.array(train.has_cactus)
# X_train = []

# for name in train.id:

#     X_train.append(imread('../input/train/train/'+name))
# X_test = []

# for name in test.id:

#     X_test.append(imread('../input/test/test/'+name))
from keras.applications import resnet50
# base_model = resnet50.ResNet50(include_top=False, weights='imagenet')
# X_train=np.array(X_train)

# X_test=np.array(X_test)
# X_train = resnet50.preprocess_input(X_train)

# X_test = resnet50.preprocess_input(X_test)
from keras.layers import GlobalAveragePooling2D,Dense,Dropout

from keras.models import Model
# x = base_model.output

# x = GlobalAveragePooling2D()(x)

# x = Dense(1024, activation='relu')(x)

# x =Dropout(0,5)(x)

# predictions = Dense(1, activation='sigmoid')(x)

# model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
red =ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
# history=model.fit(X_train, y_train, epochs=30,batch_size=20,validation_split=0.3,verbose=2,callbacks=[red,early_stopping])
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
pre=model.predict(X_test)[:,0]
from skimage.io import imshow


# fig = plt.figure(figsize=(40, 40))

# for i,j in enumerate(X_test[:25]):

#     str1=''

#     if pre[i] >= 0.5: 

#         str1=('I am {:.2%} sure is it has it'.format(pre[i]))

#     else: 

#         str1=('I am {:.2%} sure this is  it not has it '.format(1-pre[i]))

#     ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])    

#     print(j.shape)

#     plt.imshow(j)

#     ax.set_title(str1)
test.to_csv('submission.csv', index=False)

test.head()