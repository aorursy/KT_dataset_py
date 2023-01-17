import numpy as np # linear algebra

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers.normalization import BatchNormalization

import random

import PIL.Image

import cv2

import os

import xml.etree.ElementTree as ET

from IPython.display import SVG

imgs = []

breeds = []

breed_names = []

category_cnt = -1

counter = 0

size = 227
#Taken from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation 

#This function crops an image to fit the dog

def crop(breed_name, dog, data_dir):

  img = plt.imread(data_dir + 'images/Images/' + breed_name + '/' + dog + '.jpg')

  tree = ET.parse(data_dir + 'annotations/Annotation/' + breed_name + '/' + dog)

  xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)

  xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)

  ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)

  ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)

  img = img[ymin:ymax, xmin:xmax, :]

  return img

 
#Here we want to loop through our data and store the images

data_dir = '/kaggle/input/'

for dirname, _, filenames in os.walk(data_dir + 'images/Images'):

    category_cnt += 1

    file_cnt = 0

    if category_cnt == 20: #We only want 20 breeds because of limitations

        break

    for filename in filenames: #iterate through all files

        path = os.path.join(dirname, filename)

        breed = dirname.split("/")[5]

        dog = os.listdir(data_dir + 'annotations/Annotation/' + breed)[file_cnt]

        file_cnt += 1

        img = crop(breed, dog, data_dir) #crop image to fit dog

        breeds.append(category_cnt)

        breed_names.append(dirname.split("-")[1])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_array = PIL.Image.fromarray(img, 'RGB')

        img = img_array.resize((size,size)) #resize image

        imgs.append(np.array(img)) #store pixel array

        counter += 1

 

#Taken from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation

#Print random examples of dogs being cropped

breed_list = os.listdir(data_dir + 'images/Images/')

plt.figure(figsize=(20, 20))

for i in range(4):

  plt.subplot(421 + (i*2))

  breed = np.random.choice(breed_list)

  dog = np.random.choice(os.listdir(data_dir + 'annotations/Annotation/' + breed))

  img = plt.imread(data_dir + 'images/Images/' + breed + '/' + dog + '.jpg')

  plt.imshow(img)  

  tree = ET.parse(data_dir + 'annotations/Annotation/' + breed + '/' + dog)

  xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)

  xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)

  ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)

  ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)

  plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin])

  crop_img = crop(breed, dog, data_dir)

  plt.subplot(422 + (i*2))

  plt.imshow(crop_img)

 
#Taken from https://www.kaggle.com/msripooja/dog-images-classification-using-keras-alexnet

#Here we want to normalize our data

imgs = np.array(imgs) #Get pixel array

breeds = np.array(breeds)

imgs = imgs.astype(np.float32)

imgs = imgs / 255 #Normalize pixels

breeds = breeds.astype(np.int32)

#Split to train and test

x_train, x_test, y_train, y_test = train_test_split(imgs, breeds, test_size = 0.3, random_state = 43)



#Plot random dog images

fig,ax=plt.subplots(3,3)

fig.set_size_inches(15,15)

for i in range(3):

    for j in range (3):

        k=random.randint(0,len(breed_names))

        ax[i,j].imshow(imgs[k])

        ax[i,j].set_title('Dog: '+ breed_names[k])

       

plt.tight_layout()
#Taken from https://www.kaggle.com/msripooja/dog-images-classification-using-keras-alexnet

#Define our alexnet with the following architecture https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png    

def alex_net(dropout_rate=0.4):

    model = Sequential()

 

    model.add(Conv2D(filters=96,

                     kernel_size=(11, 11),

                     strides=(4, 4),

                     padding="valid",

                     activation="relu",

                    input_shape=(227, 227, 3)))

 

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())

 

    model.add(Conv2D(filters=256,

                     kernel_size=(5, 5),

                     strides=(1, 1),

                     padding="valid",

                     activation="relu"))

 

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())

 

    model.add(Conv2D(filters=384,

                     kernel_size=(3, 3),

                     strides=(1, 1),

                     padding="valid",

                     activation="relu"))

 

    model.add(Conv2D(filters=384,

                     kernel_size=(3, 3),

                     strides=(1, 1),

                     padding="valid",

                     activation="relu"))

 

    model.add(Conv2D(filters=256,

                    kernel_size=(3, 3),

                    strides=(1, 1),

                    padding="valid",

                    activation="relu"))

 

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(BatchNormalization())

 

    model.add(Flatten())

 

    model.add(Dense(4096,input_shape=(size,size,3),activation="relu"))

    model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

 

    model.add(Dense(4096,input_shape=(size,size,3),activation="relu"))

    model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

 

    model.add(Dense(1000,activation="relu"))

    model.add(Dropout(dropout_rate))

    model.add(BatchNormalization())

 

    model.add(Dense(20,activation="softmax"))

    model.summary()

 

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",

                  metrics=["accuracy"])

    return model
#Taken from https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

#3-fold cross validation of our data

#Takes a very long time to run so it's commented out

"""model = KerasClassifier(build_fn=alex_net, verbose=0)

batch_sizes = [16, 32]

epochs = [70, 100]

dropout_rates = [0.4, 0.6]

grid_param = dict(batch_size=batch_sizes, epochs=epochs, dropout_rate=dropout_rates)

grid = GridSearchCV(estimator=model, param_grid=grid_param, n_jobs=1)

result = grid.fit(x_train, y_train)

 

print("Best %f using %s" % (result.best_score_, result.best_params_))

means = result.cv_results_['mean_test_score']

stds = result.cv_results_['std_test_score']

params = result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))"""

#After finding the best model, evaluate it

model = alex_net()

history = model.fit(x_train, y_train, epochs=70, batch_size=32, verbose=0, validation_data=(x_test, y_test))
#Taken from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

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
loss, accuracy = model.evaluate(x_test, y_test)

print("test loss and accuracy")

print(loss,accuracy)


