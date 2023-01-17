# v60



# Whole dataset | X Epochs
import os

import cv2

import numpy as np

import time

import gc

from blist import blist

import matplotlib.pyplot as plt

import random

import keras

from tqdm  import tqdm, tqdm_notebook



from keras import regularizers

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD,Adam

from keras.models import Sequential, load_model

from keras.callbacks import History

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.contrib.labeled_tensor import batch



from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split





O_PATH = '../input/plant-dataset/plants/Plants/Train'

PATH_TEST = "../input/plant-test/plant_test/Test"

PATH_1 = "../input/plant-0/plant_0/Plant_0/Train"

PATH_2 = "../input/plant-1/plant_1/Plant_1/Train"

PATH_3 = "../input/plant-1/plant_2/Plant_2/Train"



COLOR = 'RGB'

CATEGORIES=[]

LIMIT = 150 # max images as a limit

no_it = 9000//LIMIT # max number of images in a folder

VESRION = '_v60'

model_name = 'plants'+VESRION+'.model'

SIZE=100

# os.mkdir('plts')

acc_files = 0
# os.remove('plants_v31.model') #Refreshing the data

#THE MODEL



def create_model(x_train,y_train,_epochs,_batch,_split,retrain):



    start_main = time.time()



    print(x_train.shape)

    if retrain:

        model = load_model(model_name)

        os.remove(model_name)

    else:

        model = Sequential()



        

        model = Sequential()  

        

        model.add(Conv2D(64, (4, 4), padding='same', input_shape=(SIZE, SIZE, 3)))        

        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3),strides=2))



        model.add(Conv2D(64, (3, 3)))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        

        model.add(Conv2D(32, (3, 3)))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.4))



        model.add(Flatten())

        model.add(Dense(60))

        model.add(Activation('relu'))

        model.add(Dropout(0.5))

        # Output Layer

        model.add(Dense(38))

        model.add(Activation('softmax'))

        

        # Output Layer

        model.summary()



        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        

    

    history = model.fit(x_train,y_train,epochs=_epochs, batch_size=_batch,validation_split=_split,verbose=1)

    model.save(model_name)

    

#     layer_outputs = [layer.output for layer in model.layers]

    

#     activation_model = Model(inputs=model.input, outputs=layer_outputs)

#     activations = activation_model.predict(x_train[5].reshape(-1,200,200,3))

    

#     for i in range(7,9):

#         print(i)

#         display_activation(activations, 4, 4, i)

    

    stop_main = time.time()

    print("Model "+model_name+" created in {}".format(stop_main-start_main))    

    

#     plt.plot(history.history['loss'])

#     plt.plot(history.history['val_loss'])

#     plt.title('Model loss')

#     plt.ylabel('Loss')

#     plt.xlabel('Epoch')

#     plt.legend(['Train', 'Test'], loc='upper left')

#     plt.show()

                  

#     plt.plot(history.history['acc'])

#     plt.plot(history.history['val_acc'])

#     plt.title('Model accuracy')

#     plt.ylabel('Accuracy')

#     plt.xlabel('Epoch')

#     plt.legend(['Train', 'Test'], loc='upper left')

#     plt.show()

    

#     datagen = None    

#     del datagen

    

    return model
# mod = create_model(train_x,train_y,5,64,0.3,create)

# test(mod,0)











# print(len(CATEGORIES))

# model = None

# datagen = None

# del datagen    

# del model

# history = create_model(train_x,train_y,22,32,0.3,False)

# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.title('Model loss')

# plt.ylabel('Loss')

# plt.xlabel('Epoch')

# plt.legend(['Train', 'Test'], loc='upper left')

# plt.show()





# from keras.models import Model



 

# def display_activation(activations, col_size, row_size, act_index): 

#     activation = activations[act_index]

#     activation_index=0

#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*5,col_size*3))

#     for row in range(0,row_size):

#         for col in range(0,col_size):

#             ax[row][col].imshow(activation[0, :, :, activation_index])

#             activation_index += 1

            

        
# plt.imshow(train_x[5]);
def create_testing_data(path,color,_shuffle,noBatch,categ):

    x = blist([])

    y = blist([])

    main_path=os.listdir(path)

    main_path.sort()

    total_categories=0

    for category in tqdm_notebook(main_path):

#         if "Potato" in category:

#             asdsac=0            

#         else:

#             continue

            

        if categ == 0: 

            CATEGORIES.append(category)

        

        new_path = os.path.join(path,category)

        i=0

        cat = CATEGORIES.index(category)

        k=0

        images_path = os.listdir(new_path)

        images_path.sort()

        if len(images_path)<(LIMIT*noBatch):

            continue

        total_categories+=1

        for img in images_path[int(LIMIT)*noBatch:(int(LIMIT)*(noBatch+1))]:

            try:                                

                if color == 'RGB':

                    image = cv2.imread(os.path.join(new_path,img))

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                elif color == 'GRAYSCALE':

                    image = cv2.imread(os.path.join(new_path,img),cv2.IMREAD_GRAYSCALE)

                else:

                    image = cv2.imread(os.path.join(new_path,img))

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                img = None

                del img

                image = cv2.resize(image,(SIZE,SIZE))             

                image=image/255.0

                x.append(image)

                y.append(cat)

                

                if k>=LIMIT:

                    break

                else:

                    k+=1

            except Exception as e:

                print("Error: ",e)

      

    if _shuffle:

        tqdm.write(" !!! STARTING TO SHUFFLE !!! ")

        x,y = shuffle(x,y)

    print(total_categories)  

    return x,y
# def print_conf_matrix(test,predict):

#     print(confusion_matrix(test, predict, labels=range(len(CATEGORIES))))
# def test(mod,files):

#     x_test, y_test = create_testing_data(PATH_TEST,COLOR,True,0)



#     x_test = np.asarray(x_test)

#     x_test = x_test.reshape(-1,SIZE,SIZE,3)

    

#     res=mod.predict(x_test)

    

#     #Creating a dictionary

#     CAT = CATEGORIES

#     Stats = None

#     del Stats

#     Stats = dict()

    

#     for x in CAT:

#         Stats[x] = [0,0]



#     no_correct = 0

#     no_wrong = 0

#     y_pred=[]

#     for i in range(len(res)):

#         y_pred.append(np.argmax(res[i]))

#         if y_test[i] == np.argmax(res[i]):    

#             Stats[CAT[y_test[i]]][0] = Stats[CAT[y_test[i]]][0] + 1

#             no_correct+=1

#         else:

#             Stats[CAT[y_test[i]]][1] = Stats[CAT[y_test[i]]][1] + 1

#             no_wrong+=1

#     acc = (no_correct/(no_correct+no_wrong))*100

#     print("Correct: {} from {} ({})".format(no_correct,(no_correct+no_wrong),acc))

#     print_conf_matrix(y_test,y_pred)

    

#     x_test =  None

#     del x_test

#     y_test = None

#     del y_test

    

#     res = None

#     del res

#     gc.collect()

train_x = None

train_y = None

del train_x

del train_y

gc.collect()

create=False

plot=0

CATEGORIES=[]

init_categ = 0

for z in range(3):

    if z == 0:

        PATH = PATH_1

        

    elif z == 1:

        PATH = PATH_2

        

    elif z == 2:        

        PATH = PATH_3

        

    for i in range(no_it):

        

        train_x, train_y= create_testing_data(PATH,COLOR,True,i,init_categ)

        train_x = np.asarray(train_x)

        train_x = np.reshape(train_x,(-1,SIZE,SIZE,3))

#         break

#         mod = create_model(train_x,train_y,50,32,0.3,create)

        mod = create_model(train_x,train_y,30,32,0.3,create)

    

        train_x = None

        train_y = None

        del train_x

        del train_y

        gc.collect()

#         test(mod,acc_files)

        

        create=True

    

        gc.collect()

    init_categ=1

#     break
x_test, y_test = create_testing_data(PATH_TEST,COLOR,True,0)



x_test = np.asarray(x_test)

x_test = x_test.reshape(-1,SIZE,SIZE,3)

res=mod.predict(x_test)

 
CAT = CATEGORIES

#Creating a dictionary

Stats = None

del Stats

Stats = dict()

for x in CAT:

    Stats[x] = [0,0]

    

no_correct = 0

no_wrong = 0

y_pred = []

for i in range(len(res)):

    y_pred.append(np.argmax(res[i]))

    if y_test[i] == np.argmax(res[i]):    

        Stats[CAT[y_test[i]]][0] = Stats[CAT[y_test[i]]][0] + 1

        no_correct+=1

    else:

        Stats[CAT[y_test[i]]][1] = Stats[CAT[y_test[i]]][1] + 1

        no_wrong+=1



print("Correct: {} from {}".format(no_correct,no_correct+no_wrong))



print(confusion_matrix(y_test, y_pred, labels=range(len(CATEGORIES))))



mod.save(model_name)
