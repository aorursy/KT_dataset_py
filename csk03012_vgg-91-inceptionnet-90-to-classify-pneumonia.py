import pandas as pd
import numpy as np
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split

print(tf.__version__)
batch_size = 128
epochs = 35
image_size = (300,300)
test_size = 0.2
training_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/train/*/*')
validation_images = tf.io.gfile.glob('../input/chest-xray-pneumonia/chest_xray/val/*/*')

print(f'Before division of 80:20')
print(f'Total number of training images = {len(training_images)}')
print(f'Total number of validation images = {len(validation_images)}\n')


total_files = training_images
total_files.extend(validation_images)
print(f'Total number of images : training_images + validation_images = {len(total_files)}\n')

train_images, val_images = train_test_split(total_files, test_size = test_size)
print(f'After division of 80:20')
print(f'Total number of training images = {len(train_images)}')
print(f'Total number of validation images = {len(val_images)}')

count_normal = len([x for x in train_images if "NORMAL" in x])
print(f'Normal images count in training set: {count_normal}')

count_pneumonia = len([x for x in train_images if "PNEUMONIA" in x])
print(f'Pneumonia images count in training set: {count_pneumonia}')

count_array = []
count_array += ['positive']*count_pneumonia
count_array += ['negative']*count_normal

sns.set_style('ticks')
sns.countplot(count_array)
tf.io.gfile.makedirs('/kaggle/working/val_dataset/negative/')
tf.io.gfile.makedirs('/kaggle/working/val_dataset/positive/')
tf.io.gfile.makedirs('/kaggle/working/train_dataset/negative/')
tf.io.gfile.makedirs('/kaggle/working/train_dataset/positive/')

for ele in train_images:
    parts_of_path = ele.split('/')

    if 'PNEUMONIA' == parts_of_path[-2]:
        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/train_dataset/positive/' +  parts_of_path[-1])
    else:
        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/train_dataset/negative/' +  parts_of_path[-1])
for ele in val_images:
    parts_of_path = ele.split('/')

    if 'PNEUMONIA' == parts_of_path[-2]:
        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/val_dataset/positive/' +  parts_of_path[-1])
    else:
        tf.io.gfile.copy(src = ele, dst = '/kaggle/working/val_dataset/negative/' +  parts_of_path[-1])
train_datagen = ImageDataGenerator(rescale = 1/255,
                                 rotation_range = 30,
                                 zoom_range = 0.2,
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1)
val_datagen = ImageDataGenerator(rescale = 1/255)
                                

train_generator = train_datagen.flow_from_directory(
    '/kaggle/working/train_dataset/',
    target_size = image_size,
    batch_size = batch_size ,
    class_mode = 'binary'
)

validation_generator = val_datagen.flow_from_directory(
    '/kaggle/working/val_dataset/',
    target_size = image_size,
    batch_size = batch_size ,
    class_mode = 'binary'
)

eval_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = eval_datagen.flow_from_directory(
    '../input/chest-xray-pneumonia/chest_xray/test',
    target_size = image_size,
    batch_size = batch_size , 
    class_mode = 'binary'
)

initial_bias = np.log([count_pneumonia/count_normal])
initial_bias
weight_for_0 = (1 / count_normal)*(len(train_images))/2.0 
weight_for_1 = (1 / count_pneumonia)*(len(train_images))/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
base_model1 = tf.keras.applications.VGG16(input_shape=(300, 300, 3),include_top=False, weights='imagenet')
base_model1.trainable = False


model1 = tf.keras.Sequential([
        base_model1,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
        ])

model1.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])
model1.summary()
print(len( base_model1.layers))
checkpoint_cb1 = tf.keras.callbacks.ModelCheckpoint("model1_vgg.h5",
                                                    save_best_only=True)

early_stopping_cb1 = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', patience=20, mode = 'min',restore_best_weights=True)
history1 = model1.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = epochs,
    validation_data = validation_generator,
    class_weight = class_weight,
    callbacks = [checkpoint_cb1, early_stopping_cb1]
)
figure, axis = plt.subplots(1, 2, figsize=(18,5))
axis = axis.ravel()

for i,element in enumerate(['accuracy', 'loss']):
    axis[i].plot(history1.history[element])
    axis[i].plot(history1.history['val_' + element])
    axis[i].set_title('Model {}'.format(element))
    axis[i].set_xlabel('epochs')
    axis[i].set_ylabel(element)
    axis[i].legend(['train', 'val'])

eval_result1 = model1.evaluate_generator(test_generator, 624)
print('loss rate at evaluation data :', eval_result1[0])
print('accuracy rate at evaluation data :', eval_result1[1])
vgg_model = tf.keras.models.load_model('/kaggle/working/model1_vgg.h5')

wrong_predicted_image = [[],[]]
correct_predicted_image = [[],[]]
i = 0
while i< 5 and len(wrong_predicted_image[0]) < 6:
    j = 0
    while j < 128 and len(wrong_predicted_image[0]) < 6:
        
        image_array = (test_generator[i][0][j]).reshape(1,300,300,3)
        
        prediction = vgg_model.predict(image_array)
        
        if int(round(prediction[0][0])) != test_generator[i][1][j]:
            wrong_predicted_image[0].append(image_array)
            wrong_predicted_image[1].append(int(round(prediction[0][0])))
            
        elif len(correct_predicted_image[0]) < 6:
            correct_predicted_image[0].append(image_array)
            correct_predicted_image[1].append(int(round(prediction[0][0])))
#         print(len(correct_predicted_image[0]),len(wrong_predicted_image[0]))  
        j += 1
        
    i += 1
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 22 ,4
fig, ax = plt.subplots(1,6)

i = 0
for ele in wrong_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'wrong_prediction_by_model --- {wrong_predicted_image[1]}')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 22 ,4
fig, ax = plt.subplots(1,6)

i = 0
for ele in correct_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'correct_prediction_by_model --- {correct_predicted_image[1]}')
base_model2 = tf.keras.applications.InceptionV3(input_shape=(300, 300, 3),include_top=False, weights='imagenet')

for layers in base_model2.layers[:200]:
    layers.trainable = False

model2 = tf.keras.Sequential([
        base_model2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid) 
        ])

model2.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

model2.summary()
len(base_model2.layers)
checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint("model1_inceptionNet.h5",
                                                    save_best_only=True)

early_stopping_cb2 = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', patience=20, mode = 'min',restore_best_weights=True)
                                                     
history2 = model2.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = epochs,
    validation_data = validation_generator,
    class_weight = class_weight,
    callbacks = [checkpoint_cb2, early_stopping_cb2]    
)
figure, axis = plt.subplots(1, 2, figsize=(18,5))
axis = axis.ravel()

for i,element in enumerate(['accuracy', 'loss']):
    axis[i].plot(history2.history[element])
    axis[i].plot(history2.history['val_' + element])
    axis[i].set_title('Model {}'.format(element))
    axis[i].set_xlabel('epochs')
    axis[i].set_ylabel(element)
    axis[i].legend(['train', 'val'])
eval_result2 = model2.evaluate_generator(test_generator, 624)
print('loss rate at evaluation data :', eval_result2[0])
print('accuracy rate at evaluation data :', eval_result2[1])
Inception_model = tf.keras.models.load_model('/kaggle/working/model1_inceptionNet.h5')

wrong_predicted_image = [[],[]]
correct_predicted_image = [[],[]]
i = 0
while i< 5 and len(wrong_predicted_image[0]) < 6:
    j = 0
    while j < 128 and len(wrong_predicted_image[0]) < 6:
        
        image_array = (test_generator[i][0][j]).reshape(1,300,300,3)
        
        prediction = Inception_model.predict(image_array)
        
        if int(round(prediction[0][0])) != test_generator[i][1][j]:
            wrong_predicted_image[0].append(image_array)
            wrong_predicted_image[1].append(int(round(prediction[0][0])))
            
        elif len(correct_predicted_image[0]) < 6:
            correct_predicted_image[0].append(image_array)
            correct_predicted_image[1].append(int(round(prediction[0][0])))
#         print(len(correct_predicted_image[0]),len(wrong_predicted_image[0]))
        j += 1
        
    i += 1
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 22 ,4
fig, ax = plt.subplots(1,6)

i = 0
for ele in wrong_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'wrong_prediction_by_model --- {wrong_predicted_image[1]}')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 22 ,4
fig, ax = plt.subplots(1,6)

i = 0
for ele in correct_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'correct_prediction_by_model --- {correct_predicted_image[1]}')
base_model3 = tf.keras.applications.ResNet50(input_shape=(300, 300, 3),include_top=False, weights='imagenet')
# base_model3.trainable = False
for layers in base_model3.layers[:100]:
    layers.trainable = False

model3 = tf.keras.Sequential([
        base_model3,
        tf.keras.layers.GlobalAveragePooling2D(),
#          tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation=tf.nn.relu),
#          tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(512, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1,activation=tf.nn.sigmoid),
        ])

model3.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])
model3.summary()
len(base_model3.layers)
checkpoint_cb3= tf.keras.callbacks.ModelCheckpoint("model3_resnet.h5",
                                                    save_best_only=True)

early_stopping_cb3 = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', patience=20, mode = 'min',restore_best_weights=True)
                                                     
history3 = model3.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = epochs,
    validation_data = validation_generator,
    class_weight = class_weight,
    callbacks = [checkpoint_cb3, early_stopping_cb3] 
)
figure, axis = plt.subplots(1, 2, figsize=(18,5))
axis = axis.ravel()

for i,element in enumerate(['accuracy', 'loss']):
    axis[i].plot(history3.history[element])
    axis[i].plot(history3.history['val_' + element])
    axis[i].set_title('Model {}'.format(element))
    axis[i].set_xlabel('epochs')
    axis[i].set_ylabel(element)
    axis[i].legend(['train', 'val'])
eval_result3 = model3.evaluate_generator(test_generator, 624)
print('loss rate at evaluation data :', eval_result3[0])
print('accuracy rate at evaluation data :', eval_result3[1])
Residual_model = tf.keras.models.load_model('/kaggle/working/model3_resnet.h5')

wrong_predicted_image = [[],[]]
correct_predicted_image = [[],[]]
i = 0
while i< 5 and len(wrong_predicted_image[0]) < 6:
    j = 0
    while j < 128 and len(wrong_predicted_image[0]) < 6:
        
        image_array = (test_generator[i][0][j]).reshape(1,300,300,3)
        
        prediction = Residual_model.predict(image_array)
        
        if int(round(prediction[0][0])) != test_generator[i][1][j]:
            wrong_predicted_image[0].append(image_array)
            wrong_predicted_image[1].append(int(round(prediction[0][0])))
            
        elif len(correct_predicted_image[0]) < 6:
            correct_predicted_image[0].append(image_array)
            correct_predicted_image[1].append(int(round(prediction[0][0])))
#         print(len(correct_predicted_image[0]),len(wrong_predicted_image[0]))
        j += 1
        
    i += 1
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 22 ,4
fig, ax = plt.subplots(1,6)

i = 0
for ele in wrong_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'wrong_prediction_by_model --- {wrong_predicted_image[1]}')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 11 ,4
fig, ax = plt.subplots(1,2)

i = 0
for ele in correct_predicted_image[0]:
    image = tf.keras.preprocessing.image.array_to_img(ele.reshape(300,300,3))
    ax[i].imshow(image)
    i += 1

print(f'correct_prediction_by_model --- {correct_predicted_image[1]}')
tf.io.gfile.rmtree('/kaggle/working/val_dataset/')
tf.io.gfile.rmtree('/kaggle/working/train_dataset/')