import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from keras.models import Sequential #Build empty model structure

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization

#Import layers from keras library 

from  keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.utils import to_categorical #to do one hot encoding. We encode labels into binary definiton

# 6= 0000001000 

# 3= 0001000000 etc.



import matplotlib.pyplot as plt





train_path="/kaggle/input/mnist-in-csv/mnist_train.csv"

test_path= "/kaggle/input/mnist-in-csv/mnist_test.csv"



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

# filter warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import os

print(os.listdir("../input/mnist-in-csv"))



# Any results you write to the current directory are saved as output.
def load_and_process(data_path):

    data=pd.read_csv(data_path)

    data=data.as_matrix() #convert dataframe into array

    np.random.shuffle(data)

    #normalize the pictures

    x=data[:,1:].reshape(-1,28,28,1)/255.0 

    #get labels

    y=data[:,0].astype(np.int32) 

    #convert labels

    y=to_categorical(y,num_classes=len(set(y)))

    return x,y



x_train,y_train=load_and_process(train_path)

x_test,y_test=load_and_process(test_path)
#%visualize

index= 128 # visualize number index

vis=x_train.reshape(60000,28,28)

plt.imshow(vis[index,:,:])

plt.legend()

plt.axis("off")

plt.show()

print(np.argmax(y_train[index]))
#Create Model

numberOfClass= y_train.shape[1] #label number (0..9)



model=Sequential() #empty model



model.add(Conv2D(input_shape=(28,28,1), filters=16, kernel_size= (3,3))) #add convolution layer

model.add(BatchNormalization())

model.add(Activation("relu")) 

model.add(MaxPooling2D())



model.add(Conv2D(filters=64, kernel_size=(3,3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Conv2D(filters=128, kernel_size=(3,3)))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(units=256))

model.add(Activation("relu"))

model.add(Dropout(0.2))

model.add(Dense(units=numberOfClass))

model.add(Activation("softmax"))



model.compile(loss="categorical_crossentropy",

              optimizer="adam",

              metrics=["accuracy"]

             )



hist=model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=25, batch_size=4000)
#save your model

model.save_weights("cnn_mnist_model.h5") 
#%% Visualization

print(hist.history.keys())

plt.plot(hist.history["loss"], label="Train loss")

plt.plot(hist.history["val_loss"], label="Validation loss")

plt.legend()

plt.show()

plt.figure() #Birarada çıkmasını istemediğimiz için bu satırı yazdık

plt.plot(hist.history["acc"], label="Train acc")

plt.plot(hist.history["val_acc"], label="Validation acc")

plt.legend()

plt.show()
#%%save history

import json

with open("cnn_mnist_model.json","w") as f:

    json.dump(hist.history,f)  
#%%load history

import codecs

with codecs.open("cnn_mnist_model.json", "r", "utf-8") as f:

    h=json.loads(f.read())

plt.plot(h["loss"], label="Train loss")

plt.plot(h["val_loss"], label="Validation loss")

plt.legend()

plt.show()

plt.figure() #Birarada çıkmasını istemediğimiz için bu satırı yazdık

plt.plot(hist.history["acc"], label="Train acc")

plt.plot(hist.history["val_acc"], label="Validation acc")

plt.legend()

plt.show()