#get dataset and unzip dataset
!wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip
!unzip concrete_data_week4.zip
#import modules
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_resnet

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_vgg16

from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img = Image.open('/kaggle/working/concrete_data_week4/train/negative/09796.jpg')
plt.imshow(img)
img2 = Image.open('/kaggle/working/concrete_data_week4/train/positive/09792.jpg')
plt.imshow(img2)
#define constants
num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100
#create imagedatagenerator
data_generator_resnet = ImageDataGenerator(preprocessing_function = preprocess_resnet)
#create generator for train and validation dataset
train_generator = data_generator_resnet.flow_from_directory(
    'concrete_data_week4/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

validation_generator = data_generator_resnet.flow_from_directory(
    'concrete_data_week4/valid',
    target_size = (image_resize, image_resize),
    batch_size = 100,
    class_mode = 'categorical'  
)
#modify the resnet50 pretrained model
model = Sequential()

model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False
#view model summary
model.summary()
#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#parameters for fitting generator
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2
#fit the generator to model
fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
#ave the model
model.save('classifier_resnet_model.h5')
#save model to google drive
#model.save('/content/gdrive/My Drive/keras_models/classifier_resnet_model.h5')
#create imagedatagenerator
data_generator_vgg16 = ImageDataGenerator(preprocessing_function = preprocess_vgg16)
#create a training data generator
train_generator = data_generator_vgg16.flow_from_directory(
    'concrete_data_week4/train',
    target_size = (image_resize, image_resize),
    batch_size = batch_size_training,
    class_mode = 'categorical'
)

#create a validation data generator
validation_generator = data_generator_vgg16.flow_from_directory(
    'concrete_data_week4/valid',
    target_size = (image_resize, image_resize),
    batch_size = batch_size_validation,
    class_mode = 'categorical'
)
#modify pretrained model
model2 = Sequential()

model2.add(VGG16(
    include_top = False,
    pooling = 'avg',
    weights = 'imagenet'
))

model2.add(Dense(num_classes, activation = 'softmax'))

model2.layers[0].trainable = False
model2.summary()
#compile the model
model2.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
#initialize epochs and steps
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2
#fit the generator to model
fit_history2 = model2.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch_training,
    epochs = num_epochs,
    validation_data = validation_generator,
    validation_steps = steps_per_epoch_validation
)
#save model locally
model2.save('classifier_vgg16_model.h5')
#save model to drive
#model2.save('/content/gdrive/My Drive/keras_models/classifier_vgg16_model.h5')
#to copy from drive
#!cp /content/gdrive/My\ Drive/keras_models/classifier_resnet_model.h5 ./classifier_resnet_model.h5
#create a test data generator
data_generator = ImageDataGenerator()

test_generator = data_generator.flow_from_directory(
    'concrete_data_week4/test',
    target_size = (image_resize, image_resize),
    shuffle = False
)
model_resnet = load_model('classifier_resnet_model.h5')
#evaluate resnet model
eval_resnet = model_resnet.evaluate_generator(
    test_generator, 
    steps=None, 
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=1)
print('Resnet50 model performance:')
print('loss for test dataset is : {}'.format(eval_resnet[0]))
print('accuracy for test dataset is : {}'.format(eval_resnet[1]))
#predict using resnet model
predict_resnet = model_resnet.predict_generator(
    test_generator, 
    steps=None, 
    callbacks=None, 
    max_queue_size=10, 
    workers=1, 
    use_multiprocessing=False, 
    verbose=1)
resnet_predict_arr = []

for i in predict_resnet:
  if int(round(i[0])) == 1:
    resnet_predict_arr.append('Positive')
  else:
    resnet_predict_arr.append('Negative')

print('PRDICTION FOR RESNET')

for i in resnet_predict_arr[0:5]:
  print(i)

print()
print('Total positives: {}'.format(resnet_predict_arr.count('Positive')))
print('Total negatives: {}'.format(resnet_predict_arr.count('Negative')))
model_vgg16 = load_model('classifier_vgg16_model.h5')
#evaluate vgg16 model
eval_vgg16 = model_vgg16.evaluate_generator(
    test_generator, 
    steps=None, 
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    verbose=1)
print('VGG16 model performance:')
print('loss for test dataset is : {}'.format(eval_vgg16[0]))
print('accuracy for test dataset is : {}'.format(eval_vgg16[1]))
#predict using vgg16 model

predict_vgg16 = model_vgg16.predict_generator(
    test_generator, 
    steps=None, 
    callbacks=None, 
    max_queue_size=10, 
    workers=1, 
    use_multiprocessing=False, 
    verbose=1)
vgg16_predict_arr = []

for i in predict_vgg16:
  if int(round(i[0])) == 1:
    vgg16_predict_arr.append('Positive')
  else:
    vgg16_predict_arr.append('Negative')

print('PRDICTION FOR VGG16')

for i in vgg16_predict_arr[0:5]:
  print(i)

print()
print('Total positives: {}'.format(vgg16_predict_arr.count('Positive')))
print('Total negatives: {}'.format(vgg16_predict_arr.count('Negative')))
