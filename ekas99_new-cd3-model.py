import keras

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, Flatten

from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D

from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint

import keras.backend as K



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import ast

import time

import random

import json

import os

import shutil

import cv2
baseLabelPath = '../input/labels/'

labelFiles = sorted(os.listdir(baseLabelPath))

print(labelFiles)



annotations = []



print('Annotations ->')

with open(os.path.join(baseLabelPath, labelFiles[0])) as file:

    dictionary = ast.literal_eval(list(file)[0]) # json data

    for items in sorted(dictionary.items()):

        annotations.append((items[0], items[1]))

        

annotations = dict(annotations) 





labels = []



print('Labels ->')

with open(os.path.join(baseLabelPath, labelFiles[1])) as file:

    dictionary = ast.literal_eval(list(file)[0])

    for items in dictionary.items():

        labels.append((items[0], items[1]))

    

labels = dict(labels)
labels
cnt = Counter()



for val in annotations.values():

    cnt[val] += 1

    

total = []

    

for val in cnt.values():

    total.append(val)



print(total)
plt.bar(sorted(total), height = 30);
"""images1 = sorted(os.listdir('../input/videos/YoloImages'))

images2 = sorted(os.listdir('../input/yoloimages2/YoloImages'))

images3 = sorted(os.listdir('../input/yoloimages3/YoloImages'))

print(len(images1) + len(images2) + len(images3))



cntLabels = Counter()

for index, image in enumerate(images1):

    imageList = image.split('_')

    file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

    cntLabels[annotations[file]] += 1

    

for index, image in enumerate(images2):

    imageList = image.split('_')

    file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

    cntLabels[annotations[file]] += 1

    

for index, image in enumerate(images3):

    imageList = image.split('_')

    file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

    cntLabels[annotations[file]] += 1

    

cntLabels """
"""num = []

for key in cntLabels.keys():

    num.append((key, int(cntLabels[key] / 16)))

    

dict(sorted(num)) """
#sum(cntLabels.values())

#cntLabels

# take only 400 images of each frame
"""cnt0, cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7, cnt8, cnt9 = [], [], [], [], [], [], [], [], [], []



samples = [996, 1070, 3500, 3490, 426, 2362, 3500, 712, 3500, 3500]



def makeList(folder):

    for index, image in enumerate(folder):

        imageList = image.split('_')

        file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

        if(annotations[file] == 0 and len(cnt0) < samples[0] * 16): cnt0.append(image)

        elif(annotations[file] == 1 and len(cnt1) < samples[1] * 16): cnt1.append(image)

        elif(annotations[file] == 2 and len(cnt2) < samples[2] * 16): cnt2.append(image)

        elif(annotations[file] == 3 and len(cnt3) < samples[3] * 16): cnt3.append(image)

        elif(annotations[file] == 4 and len(cnt4) < samples[4] * 16): cnt4.append(image)

        elif(annotations[file] == 5 and len(cnt5) < samples[5] * 16): cnt5.append(image)

        elif(annotations[file] == 6 and len(cnt6) < samples[6] * 16): cnt6.append(image)

        elif(annotations[file] == 7 and len(cnt7) < samples[7] * 16): cnt7.append(image)

        elif(annotations[file] == 8 and len(cnt8) < samples[8] * 16): cnt8.append(image)

        elif(annotations[file] == 9 and len(cnt9) < samples[9] * 16): cnt9.append(image)

            

makeList(images1)

makeList(images2)

makeList(images3) """
#len(cnt0) + len(cnt1) + len(cnt2) + len(cnt3) + len(cnt4) + len(cnt5) + len(cnt6) + len(cnt7) + len(cnt8) + len(cnt9)
368896 / 16
# Images Dataset



"""now = time.time()



images = []

labels = []



cnt = cnt1 + cnt2 + cnt3 + cnt4 + cnt5 + cnt6 + cnt7 + cnt8 + cnt9



for index, image in enumerate(images1):

    if(image in cnt): 

        imageList = image.split('_')

        file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

        

        labels.append(annotations[file])

        

        img = cv2.imread(os.path.join('../input/videos/YoloImages', image))

        img = cv2.resize(img, (112, 112))

        images.append(img)

        

print(time.time() - now)        



for index, image in enumerate(images2):

    if(image in cnt): 

        imageList = image.split('_')

        file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

        

        labels.append(annotations[file])

        

        img = cv2.imread(os.path.join('../input/yoloimages2/YoloImages', image))

        img = cv2.resize(img, (112, 112))

        images.append(img)



print(time.time() - now)

        

for index, image in enumerate(images1):

    if(image in cnt): 

        imageList = image.split('_')

        file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]

        

        labels.append(annotations[file])

        

        img = cv2.imread(os.path.join('../input/yoloimages3/YoloImages', image))

        img = cv2.resize(img, (112, 112))

        images.append(img)



print(time.time() - now) """
"""img = np.array(images)

img = np.reshape(img, (int(img.shape[0] / 16), 16, 112, 112, 3))

print(img.shape) 



np.save('trainigDataImages.npy', img)

np.save('trainingDataLabels.npy', np.array(labels)) """
"""features = []

labels = []



cnt = cnt0 + cnt1 + cnt2 + cnt3 + cnt4 + cnt5 + cnt6 + cnt7 + cnt8 + cnt9



index = 0



while index < len(images1):

    counter = 0

    

    images = []

    

    if images1[index] in cnt :

        while counter < 16:

            imageList = images1[index + counter].split('_')

            file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]



            labels.append(annotations[file])

            

            img = cv2.imread(os.path.join('../input/videos/YoloImages', images1[index + counter]))

            img = cv2.resize(img, (112, 112))

            images.append(img)

            counter += 1

    

    if counter == 16:

        feature = getFeatures(C3DModel, images)

        features.append(feature)

    

    index += 16

    

index = 0



while index < len(images2):

    counter = 0

    

    images = []

    

    if images2[index] in cnt :

        while counter < 16:

            imageList = images2[index + counter].split('_')

            file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]



            labels.append(annotations[file])

            

            img = cv2.imread(os.path.join('../input/yoloimages2/YoloImages', images2[index + counter]))

            img = cv2.resize(img, (112, 112))

            images.append(img)

            counter += 1

    

    if counter == 16:

        feature = getFeatures(C3DModel, images)

        features.append(feature)

    

    index += 16

    

index = 0



while index < len(images3):

    counter = 0

    

    images = []

    

    if images3[index] in cnt :

        while counter < 16:

            imageList = images3[index + counter].split('_')

            file = imageList[0] + '_' + imageList[1] if len(imageList) == 3 else imageList[0]



            labels.append(annotations[file])

            img = cv2.imread(os.path.join('../input/yoloimages3/YoloImages', images3[index + counter]))

            img = cv2.resize(img, (112, 112))

            images.append(img)

            counter += 1

    

    if counter == 16:

        feature = getFeatures(C3DModel, images)

        features.append(feature)

    

    index += 16 """
"""features = np.array(features)

features = np.squeeze(features, axis = 1)

print(features.shape)



np.save('trainingDataImages.npy', features)

np.save('trainingDataLabels.npy', np.array(labels)) """
# Configuring np.load

np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



def loadFile(file):

    return np.load(file)
data = loadFile('../input/trainingdatac3d/trainingDataImages.npy')

labels = loadFile('../input/trainingdatac3d/trainingDataLabels.npy')
cnt = Counter()



for index, label in enumerate(labels):

    cnt[label] += 1

    

print(cnt)
labels = np.reshape(labels, (int(labels.shape[0] / 16), 16))

print(labels.shape)
# One hot encoding

trainingLabels = []

for index, label in enumerate(labels):

    tl = [0.0] * 10

    tl[np.max(label)] = 1.0

    trainingLabels.append(tl)

    

trainingLabels = np.array(trainingLabels).astype('float')



cnt = Counter()

for index, l in enumerate(trainingLabels):

    cnt[np.argmax(l)] += 1



print(f'Samples per class: {cnt}')
## Final Data stats



print(f'Data shape: {data.shape}')

print(f'Total training samples: {data.shape[0]}')

print(f'Length of features: {data.shape[1]}')

print(f'Image size: {(data.shape[2: ])}\n')



print(f'Total labels: {trainingLabels.shape}')

print(f'Total number of labels: {trainingLabels.shape[1]}')
## Metrics



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def getC3DModel():

    """ Creates model object with the sequential API:

    https://keras.io/models/sequential/

    """



    model = Sequential()

    input_shape = (16, 112, 112, 3)



    model.add(Conv3D(64, (3, 3, 3), activation='relu',

                     padding='same', name='conv1',

                     input_shape=input_shape))

    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),

                           padding='valid', name='pool1'))

    # 2nd layer group

    model.add(Conv3D(128, (3, 3, 3), activation='relu',

                     padding='same', name='conv2'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),

                           padding='valid', name='pool2'))

    # 3rd layer group

    model.add(Conv3D(256, (3, 3, 3), activation='relu',

                     padding='same', name='conv3a'))

    model.add(Conv3D(256, (3, 3, 3), activation='relu',

                     padding='same', name='conv3b'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),

                           padding='valid', name='pool3'))

    # 4th layer group

    model.add(Conv3D(512, (3, 3, 3), activation='relu',

                     padding='same', name='conv4a'))

    model.add(Conv3D(512, (3, 3, 3), activation='relu',

                     padding='same', name='conv4b'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),

                           padding='valid', name='pool4'))

    # 5th layer group

    model.add(Conv3D(512, (3, 3, 3), activation='relu',

                     padding='same', name='conv5a'))

    model.add(Conv3D(512, (3, 3, 3), activation='relu',

                     padding='same', name='conv5b'))

    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),

                           padding='valid', name='pool5'))

    model.add(Flatten(name='featuresLayer'))

    

    # FC layers group, will be trained separately

    model.add(Dense(4096, activation='relu', name='fc6'))

    model.add(Dropout(.5))

    model.add(Dense(4096, activation='relu', name='fc7'))

    model.add(Dropout(.5))

    model.add(Dense(487, activation='softmax', name='fc8'))



    optimizer = Adam(lr=1e-4)

    loss = 'categorical_crossentropy'

    

    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy', f1_m, precision_m, recall_m])

    

    return model



def getFCModel():

    model = Sequential()

    input_shape = (8192, )

    

    model.add(Dense(4096, activation='relu', name='fc6', input_shape=input_shape))

    model.add(Dropout(.5))

    model.add(Dense(4096, activation='relu', name='fc7'))

    model.add(Dropout(.5))

    model.add(Dense(10, activation='softmax', name='fc8'))



    optimizer = Adam(lr=1e-4)

    loss = 'categorical_crossentropy'

    

    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy', f1_m, precision_m, recall_m])

    

    return model



def getFeatures(model, img):

    layer = ['featuresLayer']



    img = np.reshape(img, (1, 16, 112, 112, 3)) # reshape for model input

    

    extractor = Model(inputs=model.input,

                  outputs=model.get_layer(layer[0]).output)



    features = extractor.predict(img)



    return features
C3DModel = getC3DModel()

C3DModel.summary()



# by_name = True

# will only load weights having same name as that of original model

C3DModel.load_weights('../input/c3d-sport1m-weights-keras-224/C3D_Sport1M_weights_keras_2.2.4.h5', by_name=True)
FCModel = getFCModel()

FCModel.load_weights('../input/c3dweights-10/Best_C3DFC_model.h5')

FCModel.summary()
class SnapshotCallbackBuilder:

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.001):

        self.T = nb_epochs

        self.M = nb_snapshots

        self.alpha_zero = init_lr



    def get_callbacks(self, model_prefix='Model'):



        callback_list = [

            ModelCheckpoint("C3DFC_model.h5",monitor='val_precision_m', 

                                   mode = 'max', save_best_only=True, verbose=1),

            swa

        ]



        return callback_list
class SWA(keras.callbacks.Callback):

    

    def __init__(self, filepath, swa_epoch):

        super(SWA, self).__init__()

        self.filepath = filepath

        self.swa_epoch = swa_epoch 

    

    def on_train_begin(self, logs=None):

        self.nb_epoch = self.params['epochs']

        print('Stochastic weight averaging selected for last {} epochs.'

              .format(self.nb_epoch - self.swa_epoch))

        

    def on_epoch_end(self, epoch, logs=None):

        

        if epoch == self.swa_epoch:

            self.swa_weights = self.model.get_weights()

            

        elif epoch > self.swa_epoch:    

            for i in range(len(self.swa_weights)):

                self.swa_weights[i] = (self.swa_weights[i] * 

                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)

        else:

            pass

    

    def on_train_end(self, logs=None):

        self.model.set_weights(self.swa_weights)

        print('Final model parameters set to stochastic weight average.')

        self.model.save_weights(self.filepath)

        print('Final stochastic averaged weights saved to file.')
X_train, X_test, y_train, y_test = train_test_split(data, trainingLabels, test_size = 0.1, shuffle = True)

print(f'Training samples: {X_train.shape[0]}')

print(f'Testing samples: {X_test.shape[0]}')
epochs = 6

batchSize = 16



swa = SWA('Best_C3DFC_model.h5',epochs - 3)

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1, init_lr=1e-5)



history = FCModel.fit(X_train, y_train, epochs = epochs,

                      verbose = 1, steps_per_epoch = X_train.shape[0] // batchSize,

                      validation_split = 0.1, validation_steps = 2,

                      callbacks = snapshot.get_callbacks())
pd.DataFrame(history.history).to_hdf("C3DFCModel_20.h5",key="history")
loss, accuracy, _, precision, recall = FCModel.evaluate(X_test, y_test, verbose=0)

print(f'Loss: {np.round(loss,4)}\nAccuracy: {100 * np.round(accuracy,4)}%\nPrecision: {np.round(precision,4)}\nRecall: {np.round(recall, 4)}')
y_pred = FCModel.predict(X_test)



predictions = []

testLabels = []



for pred in y_pred:

    predictions.append(np.argmax(pred))

    

for label in y_test:

        testLabels.append(np.where(label == 1)[0][0])



        

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        

cm = confusion_matrix(testLabels, predictions)



fig = plt.figure()

ax = fig.add_subplot(111)

matC = ax.matshow(cm)



ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()
print(classification_report(testLabels, predictions));
def getFrames(video):

    cap = cv2.VideoCapture(os.path.join('videosDataset', video))



    frames = []



    while(cap.isOpened()): 

        # Capture frame-by-frame 

        ret, frame = cap.read() 



        if ret == True: 

            # Display the resulting frame

            #frame = makeJoints(videos[0].split('.mp4')[0], frame, counter)

            frames.append(cv2.resize(frame, (112, 112)))



        else: break



    return frames
"""shutil.os.mkdir('videosDataset')



videosPath = '../input/videoszip-1-2'

for folders in os.listdir(videosPath):

    for subFolders in folders:

        realPath = os.path.join(videosPath, folders, subFolders)

        for file in os.listdir(realPath):

            shutil.copy(os.path.join(realPath, file), 'videosDataset')

            

videosPath = '../input/videoszip-3-10/'

for folders in os.listdir(videosPath):

    for subFolders in os.listdir(os.path.join(videosPath, folders)):

        realPath = os.path.join(videosPath, folders, subFolders)

        for file in os.listdir(realPath):

            shutil.copy(os.path.join(realPath, file), 'videosDataset')

            

            

videos = os.listdir('videosDataset')

videos = sorted(videos) """
testVideos = sorted(random.choices(videos, k = 15))
"""trueLabels = []

predictedLabels = []



for video in testVideos:

    now = time.time()

    

    file = video.split('.')[0].split('_')[0]

    frames = getFrames(video)

    #print(np.array(frames).shape)

    features = getFeatures(C3DModel, frames)

    

    predictedLabels.append(np.argmax(FCModel.predict(features)))

    trueLabels.append(annotations[file])

    

    print(f'{video} takes {np.round(time.time() - now, 3)} seconds') """
"""correct = 0



for index in range(len(trueLabels)):

    if(trueLabels[index] == predictedLabels[index]): correct += 1



print(f'Accuracy: {(correct / len(trueLabels)) * 100}%') """