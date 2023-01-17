!mkdir ./kaggle
from google.colab import files

files.upload()
!kaggle datasets download -d paultimothymooney/blood-cells 
!ls dataset2-master/images
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras.applications import vgg16
from keras.preprocessing import image

samples = 9957
batch_size = 16

#run predictions:
generator = image.ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)

dataset = generator.flow_from_directory(
    shuffle = True,
    batch_size = 32,
    target_size = (80, 80),
    directory = 'dataset2-master/images/TRAIN'
)

def model():
    model = Sequential()
    model.add(Conv2D(80, (3,3), strides = (1, 1), activation = 'relu', input_shape = (80, 80, 3)))
    model.add(Conv2D(64, (3,3), strides = (1, 1), activation = 'relu', input_shape = (80, 80, 3)))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(64, (3,3), strides = (1,1), activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
    
    return model

nn = model()
nn.fit_generator(dataset, steps_per_epoch = None, epochs = 30, verbose = 1)
nn.save('Model.h5')

!du -h Features.npy
from google.colab import files
files.download('Model.h5')
from keras.models import load_model
import numpy
import os
from PIL import Image

model = load_model('Model.h5')

#demo code check for EOSINOPHIL

correct = 0
wrong = 0
total = 0

for file in os.listdir('dataset2-master/images/TRAIN/EOSINOPHIL'):
  image = Image.open('dataset2-master/images/TRAIN/EOSINOPHIL/'+file)
  image = image.resize((80, 80))
  image = numpy.array(image, dtype = 'float32')
  image/=255
  image = image.reshape(1, 80, 80, 3)
  prediction = model.predict(image)
  #print(numpy.argmax(prediction))
  if numpy.argmax(prediction) == 0: correct+=1
  else: wrong+=1
  total+=1

print('EOSINOPHIL :::: Result : ', 'Correct prediction %: ', (correct/total)*100, 'Wrong prediction : %', (wrong/total)*100)


#demo code check for LYMPHOCYTE

correct = 0
wrong = 0
total = 0

for file in os.listdir('dataset2-master/images/TRAIN/LYMPHOCYTE'):
  image = Image.open('dataset2-master/images/TRAIN/LYMPHOCYTE/'+file)
  image = image.resize((80, 80))
  image = numpy.array(image, dtype = 'float32')
  image/=255
  image = image.reshape(1, 80, 80, 3)
  prediction = model.predict(image)
  #print(numpy.argmax(prediction))
  if numpy.argmax(prediction) == 1: correct+=1
  else: wrong+=1
  total+=1

print('LYMPHOCYTE :::: Result : ', 'Correct prediction %: ', (correct/total)*100, 'Wrong prediction : %', (wrong/total)*100)

#demo code check for MONOCYTE

correct = 0
wrong = 0
total = 0

for file in os.listdir('dataset2-master/images/TRAIN/MONOCYTE'):
  image = Image.open('dataset2-master/images/TRAIN/MONOCYTE/'+file)
  image = image.resize((80, 80))
  image = numpy.array(image, dtype = 'float32')
  image/=255
  image = image.reshape(1, 80, 80, 3)
  prediction = model.predict(image)
  #print(numpy.argmax(prediction))
  if numpy.argmax(prediction) == 2: correct+=1
  else: wrong+=1
  total+=1

print('MONOCYTE :::: Result : ', 'Correct prediction %: ', (correct/total)*100, 'Wrong prediction : %', (wrong/total)*100)


#demo code check for NEUTROPHIL

correct = 0
wrong = 0
total = 0

for file in os.listdir('dataset2-master/images/TRAIN/NEUTROPHIL'):
  image = Image.open('dataset2-master/images/TRAIN/NEUTROPHIL/'+file)
  image = image.resize((80, 80))
  image = numpy.array(image, dtype = 'float32')
  image/=255
  image = image.reshape(1, 80, 80, 3)
  prediction = model.predict(image)
  #print(numpy.argmax(prediction))
  if numpy.argmax(prediction) == 3: correct+=1
  else: wrong+=1
  total+=1

print('NEUTROPHIL ::: Result : ', 'Correct prediction %: ', (correct/total)*100, 'Wrong prediction : %', (wrong/total)*100)
!ls dataset2-master/images/TRAIN