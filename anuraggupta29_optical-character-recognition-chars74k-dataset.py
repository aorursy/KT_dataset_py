#import packages
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import math
import os
from shutil import copyfile
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers import Flatten
#get the dataset for ocr
!wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
#unzip the dataset
!tar -xvzf EnglishFnt.tgz
#view an image sample
imgsample = Image.open('/kaggle/working/English/Fnt/Sample037/img037-00008.png')
npsample = np.array(imgsample)
plt.imshow(npsample)
#create train, valid and test directories
if not os.path.isdir('dataset'):
  os.mkdir('dataset')

if not os.path.isdir('dataset/train'):
  os.mkdir('dataset/train')
if not os.path.isdir('dataset/valid'):
  os.mkdir('dataset/valid')
if not os.path.isdir('dataset/test'):
  os.mkdir('dataset/test')
#make class directories inside them
for i in sorted(os.listdir('English/Fnt')):
  if not os.path.isdir('dataset/train/'+i):
    os.mkdir('dataset/train/'+i)
  if not os.path.isdir('dataset/valid/'+i):
    os.mkdir('dataset/valid/'+i)
  if not os.path.isdir('dataset/test/'+i):
    os.mkdir('dataset/test/'+i)
#split main folder to train, valid, test and copy images to the new folders
base = 'English/Fnt/Sample'

for char  in range(1, 63):
  classLen = len(os.listdir(base + str(char).zfill(3)))

  trainLen = math.floor(classLen*0.80)
  validLen = math.ceil(classLen*0.15)

  randFnt = np.random.randint(low = 1, high = classLen, size = classLen)
  randTrain = randFnt[:trainLen]
  randValid = randFnt[trainLen : trainLen+validLen]
  randTest = randFnt[trainLen+validLen :]

  for imgNo in randTrain:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/train/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

  for imgNo in randValid:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/valid/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

  for imgNo in randTest:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/test/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

#define constants
num_classes = 62
image_resize = 128
batch_size_training = 128
batch_size_validation = 64
#create data generator
data_generator = ImageDataGenerator(rescale=1.0/255.0)
#create train and valid generators
train_generator = data_generator.flow_from_directory(
    'dataset/train',
    target_size = (image_resize, image_resize),
    batch_size = batch_size_training,
    color_mode = 'grayscale',
    class_mode = 'categorical'
)

validation_generator = data_generator.flow_from_directory(
    'dataset/valid',
    target_size = (image_resize, image_resize),
    batch_size = batch_size_training,
    color_mode = 'grayscale',
    class_mode = 'categorical'
)
#view batch specifications
batchX, batchy = train_generator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
#This is the best model i could create.
#Because for other cases, the improvement was minimal compared to the additional computational cost.
#create the model
def ocrModel():
  model = Sequential()
  model.add(Conv2D(32, (4,4), strides = (1,1), activation = 'relu', input_shape = (128, 128, 1)))
  model.add(MaxPooling2D(pool_size = (4,4), strides = (2,2)))
  model.add(Conv2D(64, (4,4), strides = (1,1), activation = 'relu', input_shape = (128, 128, 1)))
  model.add(MaxPooling2D(pool_size = (4,4),strides = (2,2)))

  model.add(Flatten())

  model.add(Dense(310, activation='relu'))
  model.add(Dense(num_classes, activation = 'softmax'))

  model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

  return model
#parameters for fitting
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(train_generator)
num_epochs = 10
model = ocrModel()

#view model summary
model.summary()
#fit model
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch_training,
    epochs = num_epochs,
    validation_data = validation_generator,
    validation_steps = steps_per_epoch_validation,
    verbose = 1
)
#plot the fitting history
plt.plot(range(1,11), fit_history.history['val_accuracy'], label='valid')
plt.plot(range(1,11), fit_history.history['accuracy'], label='train')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
#define test generator
test_generator = data_generator.flow_from_directory(
    'dataset/test',
    target_size = (image_resize, image_resize),
    shuffle = False,
    color_mode='grayscale'
)
#evaluate the model
eval = model.evaluate_generator(test_generator, verbose=1)
print('Model performance:')
print('loss for test dataset is : {}'.format(eval[0]))
print('accuracy for test dataset is : {}'.format(eval[1]))
#connect to your google drive
#from google.colab import drive 
#drive.mount('/content/gdrive')
#save model to google drive
#model.save('/content/gdrive/My Drive/keras_models/OCRmodel.h5')
#to copy from drive
#!cp /content/gdrive/My\ Drive/keras_models/classifier_resnet_model.h5 ./classifier_resnet_model.h5
#upload files from local storage
#from google.colab import files
#uploaded = files.upload()
#create a label map
classArr = [str(i) for i in range(10)]
classArr.extend([chr(i) for i in range(ord('A'), ord('Z')+1)])
classArr.extend([chr(i) for i in range(ord('a'), ord('z')+1)])
#view the text images to be detected
fig, axs = plt.subplots(1,8, figsize = (16,5))

for i in range(8): 
    image_data = Image.open('/kaggle/input/'+str(i+1)+'.png')
    axs[i].imshow(image_data)
#detect text using model
res = ''
for i in range(1,9):
  img = Image.open('/kaggle/input/'+str(i)+'.png')
  imgnp = np.array(img)
  imgnp = np.reshape(imgnp, (1,imgnp.shape[0],imgnp.shape[1], 1))
  predict = model.predict_classes(imgnp)
  res += classArr[predict[0]]

print(res)
