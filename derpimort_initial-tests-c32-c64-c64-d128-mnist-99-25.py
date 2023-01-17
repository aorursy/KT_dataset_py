# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_DIR="/kaggle/input/digit-recognizer/"



train=pd.read_csv(os.path.join(DATA_DIR,"train.csv"))

test=pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

sample=pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
train.info()

test.info()
train['label'].value_counts()
from sklearn.model_selection import train_test_split



train_X, val_X, train_Y, val_Y=train_test_split(train.iloc[:,1:], train['label'].values, test_size=0.1, random_state=169)
train_X=train_X.to_numpy().reshape((-1,28,28,1))

val_X=val_X.to_numpy().reshape((-1,28,28,1))

test=test.to_numpy().reshape((-1,28,28,1))
num_rows=28

num_cols=28

b_size = 378

n_epoch = 10

n_classes = 10
from keras_preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(

    rotation_range=15,

    rescale=1./255.,

    width_shift_range=0.10,

    height_shift_range=0.10

)



train_datagen.fit(train_X)

val_X=val_X/255.

test=test/255.
from keras.utils import to_categorical



val_labels=val_Y

train_Y=to_categorical(train_Y, num_classes=n_classes)

val_Y=to_categorical(val_Y, num_classes=n_classes)
from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D



model=Sequential([

    Conv2D(32,(3,3),activation="relu",input_shape=(num_rows,num_cols,1)),

    Conv2D(64,(3,3),activation="relu"),

    MaxPool2D((2,2)),

    Conv2D(64,(3,3),activation="relu"),

    MaxPool2D((2,2)),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.25),

    Dense(n_classes,activation="softmax")])



model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy'])
history=model.fit_generator(

            train_datagen.flow(train_X, train_Y,batch_size=b_size),

            steps_per_epoch=(train_X.shape[0]//b_size)*1.1,

            epochs=n_epoch,

            validation_data=(val_X, val_Y),

            verbose=1)
model.save("MNIST_first.h5")
model.evaluate(val_X, val_Y)
val_predictions=model.predict(val_X)

test_predictions=model.predict(test)
from sklearn.metrics import classification_report

print(classification_report(val_labels, np.argmax(val_predictions,axis=1)))
import matplotlib.pyplot as plt



pred=np.argmax(val_predictions, axis=1)

wrong_indices=np.where(np.argmax(val_predictions,axis=1)!=val_labels)[0]



for i in range(11):

    x=1

    f, axarr=plt.subplots(1,4)

    for i in wrong_indices[i*4:i*4+4]:

        axarr[x-1].imshow(val_X[i].reshape(28,28),cmap="gray")

        axarr[x-1].set_title("Predicted: "+str(pred[i])+"\nActual: "+str(val_labels[i]))

        x+=1

    plt.show(block=True)
sample['Label']=np.argmax(test_predictions,axis=1)
sample.head()
sample.to_csv("Submissions.csv", index=False)
from keras.models import load_model



model=load_model("/kaggle/input/trained-models/MNIST_first.h5")
model.summary()