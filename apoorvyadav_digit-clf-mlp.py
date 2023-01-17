# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head()
y = df['label']
df.drop(['label'],axis=1,inplace=True)
npdf = df.to_numpy()
s=npdf.shape
X = npdf.reshape((s[0],28,28))
img = X[3,:,:]
#try if we got the image correct by plotting
plt.imshow(img,cmap="gray")
X = tf.convert_to_tensor(X)
X = tf.expand_dims(X,-1)
y = tf.convert_to_tensor(y,dtype='int32')
y = tf.keras.utils.to_categorical(y,10)
print(X.shape,y.shape)
Xtrain,Xtest,ytrain,ytest = X[:36000,:,:],X[36000:,:,:],y[:36000,:],y[36000:,:]
print(Xtrain.shape)
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Softmax
model = Sequential()
model.add(Flatten())
model.add(Dense(units = 256,activation = 'relu'))
model.add(Dense(units = 128,activation = 'relu'))
model.add(Dense(units = 64,activation = 'relu'))
model.add(Dense(units = 10,activation='softmax'))
model.summary()
batch_size = 128
epochs = 20
print(Xtest.shape,ytest.shape)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics =['accuracy'])
model.fit(Xtrain,ytrain,batch_size=batch_size,epochs=epochs,validation_data=(Xtest,ytest))
testdf = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
testdf.head()
nptestdf = testdf.to_numpy()
s=nptestdf.shape
X_test = nptestdf.reshape((s[0],28,28))
y_test = model.predict(X_test)
y_test[2]
#try if we got the image correct by plotting
plt.imshow(X_test[2,:,:],cmap="gray")
y_label = tf.argmax(y_test,axis=1)
y_label[:10]
fis,axes = plt.subplots(10,1)
for i in range(10):
    axes[i].imshow(X_test[i,:,:],cmap="gray")
out_df = pd.DataFrame(y_label)
out_df['ImageId'] = out_df.index
out_df['Label'] = out_df[0]
out_df.drop(columns = [0],inplace = True)
out_df.head()
out_df.to_csv("/kaggle/working/out.csv",index=True)
