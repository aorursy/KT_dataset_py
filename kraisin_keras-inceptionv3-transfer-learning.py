import numpy as np 

import pandas as pd



from keras.applications.inception_v3 import InceptionV3 , preprocess_input , decode_predictions

from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.layers import *

from keras.optimizers import SGD , Adam

from keras.callbacks import ModelCheckpoint



import os

print(os.listdir("../input"))

print(os.listdir('../input/stanford-car-dataset-by-classes-folder/car_data/car_data/'))
train_img_path = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'

test_img_path = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data/test'
car_name_train = {}

for i in os.listdir(train_img_path):

    car_name_train[i] =  os.listdir(train_img_path+'/'+i)
car_img_list = []

car_name_list = []

car_classes = []

car_dr = []
for i in car_name_train:

    car_classes.append(i)
for i , j in enumerate(car_name_train.values()):

    for img in j :

        car_img_list.append(img)

        car_name_list.append(car_classes[i])
for i in range(len(car_name_list)):

    car_dr.append(train_img_path+'/'+car_name_list[i]+'/'+car_img_list[i])
height = 299

width = 299



'''Notice how when we initialise our base model we set include_top=False . 

This setting is important, as it means that

we won’t be keeping the Fully-Connected (FC) layers at the end of the model'''



weight_path = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = InceptionV3(weights = weight_path ,

                      include_top = False,

                     input_shape = (height , width , 3))
'''

Now we’ll need to create a data generator to actually get our data from our

folders and into Keras in an automated way. Keras provides convenient python 

generator functions for this purpose.

'''

train_data_generator = ImageDataGenerator(

        preprocessing_function = preprocess_input,

        rotation_range = 10, # data augmentation

        horizontal_flip = True,# data augmentation

        vertical_flip = True

        )



'''

the flow_from_directory function, which will use a queue to maintain a 

continuous flow of loading and preparing our images

'''

train_generator = train_data_generator.flow_from_directory(train_img_path,

                                                          target_size = (height , width), 

                                                          batch_size = 80)
base_model.summary()
'''

We start by freezing all of the base model’s layers.

We don’t want to train those layers since we are trying to leverage 

the knowledge learned by the network from the previous dataset (in this case ImageNet).

By setting the layer.trainable=False , 

we are telling Keras not to update those weights during training,

which is exactly what we want

'''

def built_finetune_model(base_model , fc_layers , n_classes):

    for layer in base_model.layers:

        layer.trainable = False

    

    x = base_model.output

    x = Flatten()(x)

    

    for unit in fc_layers:

        x = Dense(units = unit , activation = 'relu')(x)

    

    predictions = Dense(n_classes , activation = 'softmax')(x)

    

    finetune_model = Model(inputs = base_model.input , outputs =  predictions)

    

    return finetune_model

        
n_classes = 196

fc_layers = [1050 , 500]



finetune_model = built_finetune_model(base_model , 

                                     fc_layers , 

                                     n_classes)
finetune_model.summary()
epochs = 100

batch_size = 80 

num_train_images = 8144



adam = Adam(lr = 0.00001)

finetune_model.compile(adam , loss = 'categorical_crossentropy',

                         metrics = ['accuracy'])



filepath = 'Resnet50_weights1.h5'

checkpoint = ModelCheckpoint(filepath , monitor = ['acc'] , verbose = 1  , mode = 'max')

callbacks_list = [checkpoint]



history = finetune_model.fit_generator(train_generator , epochs = epochs , 

                                      workers = 8 , steps_per_epoch = num_train_images//batch_size ,

                                      shuffle = True , callbacks = callbacks_list)
import matplotlib.pyplot as plt

def plot_loss(history):

    acc = history.history['acc']

    loss = history.history['loss']

    epoch = range(len(loss))

    

    plt.style.use('fivethirtyeight')

    plt.figure(1 , figsize = (15 , 7))

    plt.subplot(1 , 2  , 1)

    plt.plot(epoch , loss, 'r-' , alpha = 0.5)

    plt.plot(epoch , loss, 'ro')

    plt.title('epoch vs loss')

    plt.subplot(1 , 2 , 2)

    plt.plot(epoch , acc, 'g-' , alpha = 0.5)

    plt.plot(epoch , acc, 'go')

    plt.title('epoch vs accuracy')

    plt.show()
plot_loss(history)
test_img = []

ground_truth = []

for brand in os.listdir(test_img_path):

    n = 0

    for img in os.listdir(test_img_path+'/'+brand):

        n += 1

        if n == 5:

            break

        i = image.load_img(test_img_path+'/'+brand+'/'+img , target_size = (299,299))

        i = image.img_to_array(i)

        i = preprocess_input(i)

        ground_truth.append(brand)

        test_img.append(i)    
len(test_img)
test_img = np.array(test_img)
ground_truth = np.array(ground_truth)
pred = finetune_model.predict(test_img)
pred_class = []

for i in range(len(pred)):

    pred_class.append(np.argmax(pred[i]))
pred_class = np.array(pred_class)
n = 0

for i in range(16):

    n += 1

    r = np.random.randint(0 , 784 , 1)

    

    plt.figure(n , figsize = (15 , 9))

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(test_img[r][0])

    plt.title('Ground Truth : {} | Predicted : {}'.format(ground_truth[r] , car_classes[pred_class[r][0]]))

    plt.xticks([]) , plt.yticks([])

    

    if n == 16:

        break

        

plt.show()