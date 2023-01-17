# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



import numpy as np 

import pandas as pd 

import cv2 as cv 

import os

from tqdm import tqdm 



import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint





########## Extracting the input training data ################################################



train_path = '../input/train/'



test_path = '../input/test1/'



train_data = []

train_label = []

#PATH = os.path.join(train_path)

#print(PATH)



for file in tqdm(os.listdir(train_path)): 

    img = cv.imread(os.path.join(train_path,file),cv.IMREAD_GRAYSCALE)

    new_img = cv.resize(img,(96,96))

#    plt.imshow(img)

#    plt.show()



    if file.startswith('cat'):

        train_label.append(0)

    elif file.startswith('dog'):

        train_label.append(1)

    try: 

        train_data.append(new_img/255)

    except:

        train_label = train_label[:len(train_label)-1]



print(len(np.array(train_label)))

#print(np.array(train_data))



training_data = np.array(train_data)



training_label = np.array(train_label)





training_data = training_data.reshape(training_data.shape[0],training_data.shape[1],training_data.shape[2],1)



print(training_data.shape)

print(training_label.shape)



###############################################################

######### build the CNN model ##############################  



early_stopping_monitor = EarlyStopping(patience=2)



checkpoint = ModelCheckpoint('weights.hdf5',monitor='val_loss',save_best_only=True)



callbacks_list = [checkpoint]



model = Sequential()



input_shape  = (96,96,1)



model.add(Conv2D(filters=64,kernel_size=2,activation='relu',input_shape=(96,96,1),padding='same'))

model.add(Conv2D(filters=64,kernel_size=2,activation='relu',padding='valid',strides=2))

model.add(MaxPool2D(2))

#model.add(Flatten())



model.add(Conv2D(filters=64,kernel_size=2,activation='relu',padding='same'))

model.add(Conv2D(filters=64,kernel_size=2,activation='relu',padding='valid',strides=2))

model.add(MaxPool2D(2))



model.add(Conv2D(filters=64,kernel_size=2,activation='relu',padding='valid'))



model.add(Flatten())



model.add(Dropout(0.3))



model.add(Dense(100,activation='sigmoid'))

model.add(Dense(1,activation='sigmoid'))



#model.add(Conv2D(kernel_size=(3,3),filters=32,input_shape=input_shape,activation="relu"))

#model.add(Conv2D(kernel_size=(3,3),filters=64,activation="relu",padding="same"))

#model.add(MaxPool2D(pool_size=(5,5),strides=(2,2)))

#

#model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu"))

#model.add(Conv2D(kernel_size=(3,3),filters=5,activation="relu"))

#

#model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

#

#model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))

#

#model.add(Flatten())

#

#model.add(Dropout(0.3))



#model.add(Dense(100,activation="sigmoid"))

#model.add(Dense(1,activation="sigmoid"))





model.summary()



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



fitting_history = model.fit(training_data,training_label,validation_split=0.2,epochs=20,batch_size=10,

                    callbacks=callbacks_list)





run_data = fitting_history.history



train_loss = run_data['loss']

validate_loss = run_data['val_loss']



train_acc = run_data['acc']

validate_acc = run_data['val_acc'] 



epochs = range(1, len(train_loss)+1)



plt.plot(epochs,train_loss,color='blue',label='training')

plt.scatter(epochs,validate_loss,marker=11,color='blue',label='validation')

plt.title('Training & Validation Loss')

plt.legend()

plt.savefig('figure1.eps')

plt.close() 



plt.plot(epochs,train_acc,color='red',label='training')

plt.scatter(epochs,validate_acc,marker=11,color='red',label='validation')

plt.title('Training & Validation accuracy')

plt.legend()

plt.savefig('figure2.eps')

plt.close() 



##########################################################################################



################## prediction using the model ###################33



test_data = []



for tfile in tqdm(os.listdir(test_path)):

    test_img = cv.imread(os.path.join(test_path,tfile),cv.IMREAD_GRAYSCALE)

    try: 

        test_nimg = cv.resize(test_img,(96,96))

        test_data.append(test_nimg/255)

    except:

        print('')



#print(test_data)



testing_data = np.array(test_data) 



testing_data = testing_data.reshape(testing_data.shape[0],testing_data.shape[1],testing_data.shape[2],1)



print(testing_data.shape)



model.load_weights('weights.hdf5')



predicted_label = model.predict(testing_data)



print(predicted_label)



labels = [1 if value>0.5 else 0 for value in predicted_label]



print(labels)



# Any results you write to the current directory are saved as output.