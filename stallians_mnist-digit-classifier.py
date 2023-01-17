# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tensorflow import keras # model definition and training 

import matplotlib.pyplot as plt # for visualisation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

# df.head() # the data was loaded successfully
labeldf = df.pop('label')

digitsdf = df
digitsdf.describe()
digitsdf = digitsdf / 255.0
def plot_image(digit_data,labels,img_idx, cb=False):

    plt.figure()

    plt.imshow(digit_data.iloc[img_idx].values.reshape(28,28),cmap='binary')

    plt.xlabel("{}".format(labels.iloc[img_idx]))

    plt.xticks([])

    plt.yticks([])

    if cb: plt.colorbar()

    return plt
plot_image(digitsdf,labeldf,0,cb=True).show()
# sanity check

print(digitsdf.shape, labeldf.shape)
from sklearn.model_selection import train_test_split  # to split data

X_train, X_test,Y_train, Y_test = train_test_split(digitsdf,labeldf)
# sanity check

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(128,activation='relu',input_shape=(784,)))

model.add(Dense(10,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
model.fit(X_train,Y_train,epochs=2,batch_size=32,validation_data=(X_test,Y_test))
def predict_image(img_data,labels_data,img_idx):

    img = img_data.iloc[img_idx]

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    plabel = np.argmax(prediction)

    c='g' if plabel == labels_data.iloc[img_idx] else 'r'

       

    plt = plot_image(img_data,labels_data,img_idx)

    plt.xlabel("Prediction:{}, Was:{}".format(plabel,labels_data.iloc[img_idx]),color=c)

    return plt

    
predict_image(X_test,Y_test,2311).show()
testdata = pd.read_csv('../input/test.csv')
testdata.shape
predictions=model.predict(testdata)
pred=[np.argmax(predictions[i]) for i in range(len(predictions))] 
resultdf = pd.DataFrame.from_dict(data={"ImageId":range(1,28001),"Label":pred},dtype='int64')
resultdf.to_csv('ashish_agrawal_mnist_submission.csv',index=False)