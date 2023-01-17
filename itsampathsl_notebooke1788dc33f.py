# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# #dataDir = "../input/plant-seedlings-classification"
# dataDir = "/kaggle/input/plant-seedlings-classification"

species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
          'Loose Silky-bent', 'Maize','Scentless Mayweed', 'Shepherds Purse',
          'Small-flowered Cranesbill', 'Sugar beet']
import os
import pandas as pd

dataRootDir = '/kaggle/input/plant-seedlings-classification/'
trainDir = dataRootDir + 'train'
testDir = dataRootDir + 'test'

# Organize training files into DataFrame
trainData = []
for speciesId, sp in enumerate(species):
    for file in os.listdir(os.path.join(trainDir, sp)):
        trainData.append(['train/{}/{}'.format(sp, file), speciesId, sp])

train = pd.DataFrame(trainData, columns=['File', 'SpeciesId', 'Species'])
train.head()
import numpy as np
import matplotlib.pyplot as plt

# Plot a bar chart
classes= []
sampleCounts= []

for f in os.listdir(trainDir):
    trainClassPath = os.path.join(trainDir, f)
    if os.path.isdir(trainClassPath):
        classes.append(f)
        sampleCounts.append(len(os.listdir(trainClassPath)))

plt.rcdefaults()
fig, ax = plt.subplots()
yPos = np.arange(len(classes))
ax.barh(yPos, sampleCounts, align='center')
ax.set_yticks(yPos)
ax.set_yticklabels(classes)
ax.invert_yaxis()
ax.set_xlabel('Sample Counts')
ax.set_title('Sample Counts Per Class')

plt.show()
import random
from keras.preprocessing import image

ScaleTo = 100  # px to scale
fig = plt.figure(figsize= (10, 15))
fig.suptitle('Random Samples From Each Class', fontsize=14, y=.92, horizontalalignment='center', weight='bold')

columns = 5
rows = 12
for i in range(12):
    sampleClass= os.path.join(trainDir,classes[i])
    for j in range(1,6):
        fig.add_subplot(rows, columns, i*5+j)
        plt.axis('off')
        if j==1:
            plt.text(0.0, 0.5,str(classes[i]).replace(' ','\n'), fontsize=13, wrap=True)
            continue
        randomImage= os.path.join(sampleClass, random.choice(os.listdir(sampleClass)))
        img = image.load_img(randomImage, target_size=(ScaleTo, ScaleTo))
        img= image.img_to_array(img)
        img /= 255.
        plt.imshow(img)
        
plt.show()


# import shutil

# # Before doing this, we must address the fact that there is no validation dataset yet. We will construct a validation set using 30% of 
# # the training set. In order to maintain the same distribution, we will randomly select 30% from each class.
# # create validation set

# validationDir = './validation'

# def createValidation(validationSplit=0.3):
#     if os.path.isdir(validationDir):
#         print('Validation directory already created!')
#         print('Process Terminated')
#         return
#     os.mkdir(validationDir)
#     for f in os.listdir(trainDir):
#         trainClassPath= os.path.join(trainDir, f)
#         if os.path.isdir(trainClassPath):
#             validationClassPath= os.path.join(validationDir, f)
#             os.mkdir(validationClassPath)
#             filesToMove= int(0.3*len(os.listdir(trainClassPath)))
            
#             for i in range(filesToMove):
#                 randomImage= os.path.join(trainClassPath, random.choice(os.listdir(trainClassPath)))
#                 shutil.move(randomImage, validationClassPath)
#     print('Validation set created successfully using {:.2%} of training data'.format(validationSplit))
# createValidation()
sampleCounts= {}

for i, d in enumerate([trainDir]):

    classes= []
    sampleCounts[d]= []

    for f in os.listdir(d):
        trainClassPath= os.path.join(d, f)
        if os.path.isdir(trainClassPath):
            classes.append(f)
            sampleCounts[d].append(len(os.listdir(trainClassPath)))

    #fig, ax= plt.subplot(221+i)
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(classes))

    ax.barh(y_pos, sampleCounts[d], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Sample Counts')
    ax.set_title('{} Sample Counts Per Class'.format(d.capitalize()))

plt.show()
# Now we will attempt to remove the background from the images to see if can find a method 
# which generalizes well across all images, then this can be used to accelerate training by 
# isolating the important part of our data. The strategy will be to find upper and lower bounds 
# within a color space which will only contain the green part of the plants. We will then turn the 
# rest of the background black. In order to find the best values for these upper and lower bounds, 
# we grab random pixels from random training images from each of our 12 classes. We will then take this 
# random collection of pixels and plot it in color space i hopes that we can find upper and lower 
# bounds which cleanly seperate the green part of the plants.

import cv2
from math import sqrt, floor

def pullRandomPixels(samplesPerClass, pixelsPerSample):
    totalPixels = 12*samplesPerClass*pixelsPerSample
    randomPixels = np.zeros((totalPixels, 3), dtype=np.uint8)
    for i in range(12):
        sampleClass = os.path.join(trainDir,classes[i])
        for j in range(samplesPerClass):
            randomImage = os.path.join(sampleClass, random.choice(os.listdir(sampleClass)))
            img = cv2.imread(randomImage)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(img, (img.shape[0]*img.shape[1], 3))
            newPixels= img[np.random.randint(0, img.shape[0], pixelsPerSample)]
            
            startIndex = pixelsPerSample*(i*samplesPerClass+j)
            randomPixels[startIndex:startIndex+pixelsPerSample,:]= newPixels

    h = floor(sqrt(totalPixels))
    w = totalPixels//h
    
    randomPixels = randomPixels[np.random.choice(totalPixels, h*w, replace=False)]
    randomPixels = np.reshape(randomPixels, (h, w, 3))
    return randomPixels
    
randomPixels = pullRandomPixels(10, 50)

plt.figure()
plt.suptitle('Random Samples From Each Class', fontsize=14, horizontalalignment='center')
plt.imshow(randomPixels)
plt.show()
#plot these pixels in color space (RGB).
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

r, g, b = cv2.split(randomPixels)
fig = plt.figure(figsize=(8, 8))
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.view_init(20, 120)

pixelColors = randomPixels.reshape((np.shape(randomPixels)[0]*np.shape(randomPixels)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixelColors)
pixelColors = norm(pixelColors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixelColors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()
#  Choosing bounds of RGB values will not work due to the shape of the distribution. 
# Before resorting to more sophisticated methods to isolate these pixels, lets try a differe color space basis (HSV).

hsv_img = cv2.cvtColor(np.uint8(randomPixels), cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_img)
fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.view_init(50, 240)

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors = pixelColors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
# In HSV space, it looks like our clusters are more neatly seperable by choosing upper and lower bounds of HSV values

hsv_img = cv2.cvtColor(np.uint8(randomPixels), cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_img)
fig = plt.figure(figsize=(6,6))
axis = fig.add_subplot(1, 1, 1)

axis.scatter(h.flatten(), s.flatten(), facecolors=pixelColors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
plt.show()
# Isolate pixels with Hue values ranging from 24 to 58 and Saturation values ranging from 48 to 255.

lower_bound = (24, 58, 0)
upper_bound = (48, 255, 255)

fig= plt.figure(figsize=(10, 10))
fig.suptitle('Random Pre-Processed Image From Each Class', fontsize=14, y=.92, horizontalalignment='center', weight='bold')

for i in range(12):
    sampleClass = os.path.join(trainDir,classes[i])
    randomImage = os.path.join(sampleClass, random.choice(os.listdir(sampleClass)))
    img= cv2.imread(randomImage)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.resize(img, (150, 150))
    
    hsvImg= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsvImg, lower_bound, upper_bound)
    result = cv2.bitwise_and(img, img, mask=mask)

    fig.add_subplot(6, 4, i*2+1)
    plt.imshow(img)
    plt.axis('off')    

    fig.add_subplot(6, 4, i*2+2)
    plt.imshow(result)
    plt.axis('off')
    
plt.show()
# We will create a function to make the transformation compatible with the ImageDataGenerator object from Keras, 
# which will be using in our model.

def colorSegmentation(imgArray):
    imgArray = np.rint(imgArray)
    imgArray = imgArray.astype('uint8')
    hsvImg = cv2.cvtColor(imgArray, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsvImg, (24, 58, 0), (48, 255, 255))
    result = cv2.bitwise_and(imgArray, imgArray, mask=mask)
    result = result.astype('float64')
    return result
# Image pre-processing

testDatagen = image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range = 180,
      width_shift_range = 0.0,
      height_shift_range = 0.0,
      shear_range = 0.1,
      zoom_range = 0.1,
      horizontal_flip = True,
      vertical_flip = True,
      preprocessing_function = colorSegmentation,
      fill_mode='nearest')

testDatagen = image.ImageDataGenerator(rescale=1./255, preprocessing_function = colorSegmentation)
imgResize = 100

trainGenerator = testDatagen.flow_from_directory(
  trainDir,
  target_size=(imgResize, imgResize),
  batch_size=20,
  class_mode='categorical')

# validationGenerator = testDatagen.flow_from_directory(
#         'plant-seedlings-classification/validation',
#         target_size=(imgResize, imgResize),
#         batch_size=20,
#         class_mode='categorical')

testGenerator = testDatagen.flow_from_directory(
        testDir,
        target_size=(imgResize, imgResize),
        batch_size=20,
        class_mode='categorical',
        shuffle=False)
from prettytable import PrettyTable
numClases = 0

#get class indices and labels. calculate class weight
label_map = {}
for k, v in trainGenerator.class_indices.items():
    label_map[v] = k

classCounts = pd.Series(trainGenerator.classes).value_counts()
classWeight = {}

for i, c in classCounts.items():
    classWeight[i]= 1.0/c
    
normFactor = np.mean(list(classWeight.values()))

for k in classCounts.keys():
    classWeight[k] = classWeight[k]/normFactor

t = PrettyTable(['class_index', 'class_label', 'class_weight'])
for i in sorted(classWeight.keys()):
    t.add_row([i, label_map[i], '{:.2f}'.format(classWeight[i])])
    numClases += i
print(t)
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization

num_clases = 12
seed = 7
numpy.random.seed(seed)  # Fix seed

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(ScaleTo, ScaleTo, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_clases, activation='softmax'))

model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
import keras
from keras import models, layers, callbacks

best_cb = callbacks.ModelCheckpoint('/kaggle/working/model_best.h5', 
                                         monitor='val_loss', 
                                         verbose=1, 
                                         save_best_only=True, 
                                         save_weights_only=False, 
                                         mode='auto', 
                                         period=1)

opt = keras.optimizers.Adam(lr=0.0005, amsgrad=True)

model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit_generator(
                    trainGenerator,
                    class_weight= classWeight,
                    steps_per_epoch= 190,
                    epochs=50,
                    validation_steps= 48,
                    verbose=1,
                    use_multiprocessing=True,
                    callbacks=[best_cb])
#load best model from training
model= models.load_model('/kaggle/working/model_best.h5')

#save history
with open('model_history.pkl', 'wb') as f:
    pickle.dump(history, f)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
pred = model.predict_generator(test_generator, steps= test_generator.n, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
prediction_labels = [label_map[k] for k in predicted_class_indices]
filenames= test_generator.filenames
import csv
csvfile= open('/kaggle/working/submission.csv', 'w', newline='')
writer= csv.writer(csvfile)

headers= ['file', 'species']

writer.writerow(headers)
t = PrettyTable(headers)
for i, f, p in zip(range(len(filenames)), filenames, prediction_labels):
    writer.writerow([os.path.basename(f),p])
    if i <10:
        t.add_row([os.path.basename(f), p])
    elif i<13:
        t.add_row(['.', '.'])
csvfile.close()
print(t)