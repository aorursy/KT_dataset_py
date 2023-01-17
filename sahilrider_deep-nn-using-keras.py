# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense,Dropout, Activation,Lambda,Flatten

from keras.optimizers import Adam , RMSprop

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
test=pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
x_train=(train.ix[:,1:].values).astype('float32')

y_train=train.ix[:,0].values.astype('int32')



x_test=test.values.astype('float32')
x_train.shape
y_train.shape
x_train=x_train.reshape(x_train.shape[0],28,28)
x_train.shape
#change index to view other images

index=678

plt.imshow(x_train[index])

print('Number is',y_train[index])
x_train=x_train.reshape(x_train.shape[0],28,28,1)

x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train.shape,x_test.shape
#One Hot Encoding 

#I guess everybody knows this otherwise google



from keras.utils.np_utils import to_categorical

y_train=to_categorical(y_train,num_classes=10)
y_train.shape
#same as above to verify as if it correct

print(y_train[index])

plt.plot(y_train[index])

plt.xticks(range(10))

plt.show()
mean_px = x_train.mean().astype(np.float32)

std_px = x_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
x_train.reshape
np.random.seed(34)
model=Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(10,activation='softmax'))

#most useful function for a newbie **sobs**

model.summary()
model.compile(optimizer=RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])
from keras.preprocessing import image

gen=image.ImageDataGenerator()
X_train,X_val,Y_train,Y_val=train_test_split(x_train,y_train,test_size=0.10,random_state=34)

batches=gen.flow(X_train,Y_train,batch_size=64)

val_batches=gen.flow(X_val,Y_val,batch_size=64)
cache=model.fit_generator(batches,batches.n,nb_epoch=1,validation_data=val_batches,nb_val_samples=val_batches.n)
cache.history
model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X_train, Y_train, batch_size=64)

history=model.fit_generator(batches, batches.n, nb_epoch=1)
history.history
preds=model.predict_classes(x_test,verbose=0)
preds[0:5]
subs=pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds})

subs.to_csv("sub1.csv",index=False,header=True)