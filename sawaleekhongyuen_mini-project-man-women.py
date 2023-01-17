from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

import pandas as pd



print(os.listdir("../input/men-women-classification"))
data=[]

labels=[]



men=os.listdir("../input/men-women-classification/data/men/")

print(men)

for m in men:

    try:

        image=cv2.imread("../input/men-women-classification/data/men/"+m)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((200, 200))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



women=os.listdir("../input/men-women-classification/data/women/")

for f in women:

    try:

        image=cv2.imread("../input/men-women-classification/data/women/"+f)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((200, 200))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")
Cells=np.array(data)

labels=np.array(labels)



np.save("Cells",Cells)

np.save("labels",labels)



Cells=np.load("Cells.npy")

labels=np.load("labels.npy")



s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]



num_classes=len(np.unique(labels))

len_data=len(Cells)
(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]



x_train = x_train.astype('float32')/255 

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)



(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]



print(y_train)

print(y_test)

print(x_train.shape)

print(y_train.shape)
y_train=keras.utils.to_categorical(y_train,2)

y_test=keras.utils.to_categorical(y_test,2)
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(200,200,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=128,epochs=30)



# Save the model weights:

from keras.models import load_model

model.save('men_women.h5')
from keras.models import load_model

import matplotlib.pyplot as plt

from PIL import Image

from PIL import Image

import numpy as np

import os

import cv2



def convert_to_array(img):

    im = cv2.imread(img)

    cv_rgb =cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    plt.imshow(cv_rgb)

    plt.show()

    img_ = Image.fromarray(im, 'RGB')

    image = img_.resize((200, 200))

    return np.array(image)



def get_cell_name(label):

    if label==0:

        return "men"

    if label==1:

        return "women"

    

def predict_cell(file):

    model = load_model('men_women.h5')

    print("Predicting Type of people Image.................................")

    ar=convert_to_array(file)

    ar=ar/255

    label=1

    a=[]

    a.append(ar)

    a=np.array(a)

    score=model.predict(a,verbose=1)

    print(score)

    label_index=np.argmax(score)

    print(label_index)

    acc=np.max(score)

    Cell=get_cell_name(label_index)

    return Cell,"The people Cell is a "+Cell+" with accuracy =    "+str(acc)



# predict_cell('../input/sawalee/kiki/01.jpg')

# predict_cell('../input/sawalee/kiki/02.jpg')

# predict_cell('../input/sawalee/kiki/03.jpg')

# predict_cell('../input/sawalee/kiki/04.jpg')

# predict_cell('../input/sawalee/kiki/05.jpg')

predict_cell('../input/sawalee/kiki/06.jpg')

# predict_cell('../input/sawalee/kiki/07.jpg')

# predict_cell('../input/sawalee/kiki/08.jpg')

# predict_cell('../input/sawalee/kiki/09.jpg')

# predict_cell('../input/sawalee/kiki/10.jpg')
#Check the accuracy on Test data:

accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])