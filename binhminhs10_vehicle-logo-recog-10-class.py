# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing scikit-learn tools

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Importing standard ML set - numpy, pandas, matplotlib

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec



# Importing keras and its deep learning tools - neural network model, layers, contraints, optimizers, callbacks and utilities

from keras.models import Sequential, load_model

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.constraints import maxnorm

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import np_utils

from keras.regularizers import l2

from keras.initializers import RandomNormal, VarianceScaling
# path = "../input/testingdata/TestingData/"

# images = []

# labels= []

# brands = os.listdir(path)

# for brand in brands:

#     img = os.listdir(path+brand)

#     for i in img:

#         images.append(i)

#         labels.append(brands.index(brand))

    

# dataset, labelset = shuffle(images, labels, random_state=42)

# test_data = [dataset, labelset] 
img_x=img_y = 70

path = "../input/trainingdata/TrainingData/"

imgs = []

labels= []

brands = os.listdir(path)

print(brands)

for idcar, brand in enumerate(brands):

    img = os.listdir(path+brand)

    for i, value in enumerate(img):

        imgs.append(value)

        labels.append(idcar)
from PIL import Image

# loading all images

images = np.array([ np.array( Image.open(path+brands[labels[i]]+'/'+value).convert("RGB") ).flatten() for i, value in enumerate(imgs)], order='F', dtype='uint8')

# Mỗi ảnh có kích thước 70x70 = 2500 pixel và 3 kênh màu = 14700 pixel

print('total images: ', np.shape(images) )
dataset, labelset = shuffle(images, labels, random_state=42)

train_data = [dataset, labelset] 
# an example image

r=2434

plt.imshow(images[r].reshape(img_x, img_y, 3))

plt.title(brands[labels[r]])

plt.show()
# Training and preparing dataset

X_train, X_val, y_train, y_val = train_test_split( train_data[0], train_data[1], test_size=0.2)
# bring images back size (20778, 50, 50,3)

def ImageConvert(n, i):

    im_ex = i.reshape(n, img_x, img_y, 3)

    im_ex = im_ex.astype('float32') / 255

    # zero center data

    im_ex = np.subtract(im_ex, 0.5)

    # ...and to scale it to (-1, 1)

    im_ex = np.multiply(im_ex, 2.0)

    return im_ex

X_train = ImageConvert(X_train.shape[0], X_train)

X_val = ImageConvert(X_val.shape[0], X_val)
# Labels have to be transformed to categorical

Y_train = np_utils.to_categorical(y_train, num_classes=len(brands))

Y_val = np_utils.to_categorical(y_val, num_classes=len(brands))
# Four Conv/MaxPool blocks, a flattening layer and two dense layers at the end

def contruction(n_channels):

    model = Sequential()

    model.add(Conv2D(32, (3,3),

                     input_shape=(img_x,img_y,n_channels),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(64, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(128, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(256, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Flatten())

    

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))

    model.add(Dropout(0.5))

    

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))

    model.add(Dropout(0.5))

    

    # final activation is softmax, tuned to the number of classes/labels possible

    model.add(Dense(len(brands), activation='softmax'))

    

    # optimizer will be a stochastic gradient descent, learning rate set at 0.005

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.95, nesterov=True)

    adam = Adam()

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    return model

model = contruction(3)

# Let's look at the summary

model.summary()
# Some callbacks have to be provided to choose the best trained model

# patience set at 4 as 3 was too greedy - I observed better results after the third-worse epoch

early_stopping = EarlyStopping(patience=4, monitor='val_loss')

CNN_file = '10car_1CNN_CMCMCMCMF.h5py'



take_best_model = ModelCheckpoint(CNN_file, save_best_only=True)



# Finally for some CNN construction!

batch = 128

# there are 40 brands altogether

n_classes = len(brands)

n_epochs = 100

# images are RGB

n_channels = 3
history = model.fit(X_train, Y_train, batch_size=batch, shuffle=True, epochs=n_epochs, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping, take_best_model])
pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
import os

print(os.listdir())
# Save the weights

model.save_weights('model_weights.h5')



# Save the model architecture

with open('model_architecture.json', 'w') as f:

    f.write(model.to_json())



#model.load_weights(CNN_file)

scores = model.evaluate(X_val, Y_val) # let's look at the accuracy on the test set

print("Accuracy test: %.2f%%" % (scores[1]*100))
from sklearn.metrics import precision_recall_fscore_support as prfs



# Preparing for metrics check-up on the test set, may take a while...

Y_pred = model.predict_classes(X_val)



precision, recall, f1, support = prfs(y_val, Y_pred, average='weighted')

print("Precision: {:.2%}\nRecall: {:.2%}\nF1 score: {:.2%}\nAccuracy: {:.2%}".format(precision, recall, f1, scores[1]))


def ShowCase(cols, rows):

    fdict = {'fontsize': 24,

            'fontweight' : 'normal',

            'verticalalignment': 'baseline'}

    plt.figure(figsize=(cols * 5, rows * 4))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    c = 0

    for i in range(rows * cols):

        plt.subplot(rows, cols, i + 1)

        

        # r - randomly picked from the whole dataset

        r = np.random.randint(np.shape(images)[0])

        

        # j - predicted class for the image of index r (weird syntax, but works :)

        j = int(model.predict_classes(ImageConvert(1, images[r:r+1]), verbose=0))

        

        # increase success if predicted well

        if labels[r] == j:

            c += 1

        

        # image needs reshaping back to a 50px*50px*RGB

        plt.imshow(images[r].reshape(img_x, img_y, 3))

        

        # plt.title will show the true brand and the predicted brand

        plt.title('True brand: '+brands[labels[r]]+'\nPredicted: '+brands[j],

                  color= 'Green' if brands[labels[r]] == brands[j] else 'Red', fontdict=fdict) # Green for right, Red for wrong

        

        # no ticks

        plt.xticks(())

        plt.yticks(())

        

    # print out the success rate

    print('Success rate: {}/{} ({:.2%})'.format(c, rows*cols, c/(rows*cols)))

    

    plt.show()
# That is strictly for the showcasing, how the CNN works - ain't that bad, after all :)

ShowCase(10, 5)
img_x=img_y = 70

path = "../input/testingdata/TestingData/"

imgs_test = []

labels_test = []

brands = os.listdir(path)

for idcar, brand in enumerate(brands):

    img = os.listdir(path+brand)

    for i, value in enumerate(img):

        imgs_test.append(value)

        labels_test.append(idcar)
from PIL import Image

# loading all images

images = np.array([ np.array( Image.open(path+brands[labels_test[i]]+'/'+value).convert("RGB") ).flatten() for i, value in enumerate(imgs_test)], order='F', dtype='uint8')

# Mỗi ảnh có kích thước 70x70 = 2500 pixel và 3 kênh màu = 14700 pixel

print('total images: ', np.shape(images) )
# preparation data

dataset, labelset = shuffle(images, labels_test, random_state=42)

test_data = [dataset, labelset]

X_test = ImageConvert(test_data[0].shape[0], test_data[0])

Y_test = np_utils.to_categorical(test_data[1], num_classes=len(brands))
scores = model.evaluate(X_test, Y_test) # let's look at the accuracy on the test set

print("Accuracy test: %.2f%%" % (scores[1]*100))




from sklearn.metrics import precision_recall_fscore_support as prfs



# Preparing for metrics check-up on the test set, may take a while...

Y_pred = model.predict_classes(X_test)



precision, recall, f1, support = prfs(test_data[1], Y_pred, average='weighted')

print("Precision: {:.2%}\nRecall: {:.2%}\nF1 score: {:.2%}\nAccuracy: {:.2%}".format(precision, recall, f1, scores[1]))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns #for better and easier plots



def report_and_confusion_matrix(label, prediction):

    print("Model Report")

    print(classification_report(label, prediction))

    score = accuracy_score(label, prediction)

    print("Accuracy : "+ str(score))

    

    ####################

    fig, ax = plt.subplots(figsize=(8,8)) #setting the figure size and ax

    mtx = confusion_matrix(label, prediction)

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=True, ax=ax) #create a heatmap with the values of our confusion matrix

    plt.ylabel('true label')

    plt.xlabel('predicted label')
report_and_confusion_matrix(test_data[1], Y_pred)
# But let's check per class, too - assuming that larger datasets will be having higher metrics

precision_, recall_, f1_, support_ = prfs(test_data[1], Y_pred, average=None)
# We see that smaller sets (Lexus, Jaguar, Hyundai) have generally worse precision and recall

plt.subplots(figsize=(18,30))

x = range(len(brands))

plt.subplot(311)

plt.title('Precision per class')

plt.ylim(0.8, 1.00)

plt.bar(x, precision_, color='Red')

plt.xticks(x, brands, rotation = 90)

plt.subplot(312)

plt.title('Recall per class')

plt.ylim(0.8, 1.00)

plt.bar(x, recall_, color='Green')

plt.xticks(x, brands, rotation = 90)

plt.subplot(313)

plt.title('F1 score per class')

plt.ylim(0.8, 1.00)

plt.bar(x, f1_, color='Blue')

plt.xticks(x, brands, rotation = 90)

plt.show()