print( "hello world")
"hello world"
import os, cv2, random

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from tqdm import tqdm    #Helps in visualization

from random import shuffle #to shuffle the images 





TRAIN_DIR = '../input/paper-training-images/TrainingResized/'

TEST_DIR = '../input/paper-training-images/TrainingResized/'

IMG_SIZE = 128  



SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR) #using a subset of data as resouces as limited. 

SHORT_LIST_TEST = os.listdir(TEST_DIR)



SHORT_LIST_TRAIN
%matplotlib inline

def label_img(img): 

    if "Food container" in img: 

        return [1,0,0,0]

    elif "food tray" in img: 

        return [0,1,0,0]

    elif "paper cup" in img: 

        return [0,0,1,0]

    elif "paper plate" in img: 

        return [0,0,0,1]

    
#returns an numpy array of train and test data

def create_train_data():

    training_data = []

    for img in tqdm(SHORT_LIST_TRAIN):

        label = label_img(img)

        path = os.path.join(TRAIN_DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)

    return training_data



def process_test_data():

    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):

        path = os.path.join(TEST_DIR,img)

        img_num = img.split('.')[0]

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)

    return testing_data


labels = [] #make an empty list.

for i in SHORT_LIST_TRAIN: 

    labels.append(str(label_img(i)))



sns.countplot(labels) # show the list as a graph

plt.title('Recycling Classes')



train = create_train_data() #This is a method defined above


#import matplotlib.pyplot as plt



fig , ax = plt.subplots(3, 3, figsize=(30, 25))

for i, axis in enumerate(ax.flat):

    axis.imshow(train[i][0], cmap='gray')

    axis.set_title(f'Label:  {train[i][1]}', fontsize=20)
from tensorflow.python.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D



NUM_CLASSES = 4

RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' #importing a pretrained model

my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))

my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = True
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
my_new_model.summary()
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)

Y = np.array([i[1] for i in train])
history = my_new_model.fit(X, Y, validation_split=0.20, epochs=4, batch_size=64)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()


SHORT_LIST_TRAIN = os.listdir(TRAIN_DIR) #[0:500]

train = create_train_data()

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)

Y = np.array([i[1] for i in train])

history = my_new_model.fit(X, Y, validation_split=0.5, epochs=20, batch_size=64)



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
testing_data = []

for img in tqdm(os.listdir(TEST_DIR)[0:100]):

    path = os.path.join(TEST_DIR,img)

    img_num = img.split('.')[0]

    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

    testing_data.append([np.array(img), img_num])

    

shuffle(testing_data)    

test_data = testing_data 





fig , ax = plt.subplots(6, 4, figsize=(30, 25))

for i, axis in enumerate(ax.flat):

    axis.imshow(test_data[i][0], cmap='gray')

    img_data = test_data[i][0]

    orig = img_data

    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)

    model_out = my_new_model.predict([data])[0]    



    #axis.set(title=f'{im_pred[i].max()} => {category[im_pred[i].argmax()]}')

    axis.set_title(f'Predict: {model_out.max()} => {model_out.argmax()}', fontsize=20)