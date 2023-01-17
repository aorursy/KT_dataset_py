import numpy as np   # linear algebra

import pandas as pd   # data processing

from sklearn.datasets import load_files  # efficiently load files

import tensorflow as tf  # deep learning framework

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout

from keras.layers import Activation, Flatten

from keras.models import  Sequential

from keras import Model, optimizers

from keras.applications.vgg16 import decode_predictions

from keras.applications.resnet50 import preprocess_input

from keras.preprocessing import image  # for image preprocessing

import matplotlib.pyplot as plt # for visualization

import seaborn as sns  

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from PIL import Image   # image library

import os
# Get some examples for visualization



img1 = "../input/dogs-cats-images/dataset/training_set/dogs/dog.100.jpg"

img2 = "../input/dogs-cats-images/dataset/test_set/cats/cat.4028.jpg"

img3 = "../input/flowers-recognition/flowers/sunflower/1008566138_6927679c8a.jpg"

img4 = "../input/fruits/fruits-360/Test/Corn/178_100.jpg"

img5 = "../input/dogs-cats-images/dataset/test_set/dogs/dog.4023.jpg"

img6 = "../input/fruits/fruits-360/Training/Apple Red Delicious/101_100.jpg"

img7 = "../input/fruits/fruits-360/Training/Blueberry/103_100.jpg"

img8 = "../input/fruits/fruits-360/Training/Banana/119_100.jpg"

img9 = "../input/fruits/fruits-360/Training/Guava/116_100.jpg"

img10 = "../input/fruits/fruits-360/Training/Papaya/102_100.jpg"

img11 = "../input/fruits/fruits-360/Training/Watermelon/101_100.jpg"

unseen_data = '../input/unseen-examples/Unseen'

imgs1 = [img1, img2, img3, img4, img5]

imgs2 = [img6, img7, img8, img9, img10, img11]

disp_fruits = ['Apple', 'Blueberry', 'Banana', 'Guava', 'Papaya', 'Watermelon']
fig, ax = plt.subplots(1,6)

for i in range(6):

    ax[i].imshow(Image.open(imgs2[i]))

    ax[i].set_title(disp_fruits[i])

    ax[i].set_xticks([])

    ax[i].set_yticks([])

fig.set_size_inches(16,16)
train_path = "../input/fruits/fruits-360/Training/"

test_path = "../input/fruits/fruits-360/Test/"



# check for equality of train and test classes 

train_fv = sorted(os.listdir(train_path))

test_fv = sorted(os.listdir(test_path))

print('Train classes match Test classes:', train_fv == test_fv)
tr_count = {}

for p in train_fv:

    new_path = train_path + p 

    items = len(os.listdir(new_path))

    tr_count[p] = items

tr_data = pd.DataFrame.from_dict(tr_count, orient='index')



te_count = {}

for p in test_fv:

    new_path = test_path + p 

    items = len(os.listdir(new_path))

    te_count[p] = items

te_data = pd.DataFrame.from_dict(te_count, orient='index')



img_data = tr_data.merge(te_data, left_on=tr_data.index, right_on=te_data.index)

img_data = img_data.rename(columns={ 'key_0': 'name', '0_x':'train_imgs', '0_y': 'test_imgs'})

img_data.head()
fig = make_subplots(rows=2, cols=1)



traces = [

    go.Bar(y=img_data.train_imgs, x=img_data.name, name = 'train'),

    go.Bar(y=img_data.test_imgs, x=img_data.name, name = 'test')

]



for i in range(2):

    fig.add_trace(traces[i], i+1, 1)



fig.update_layout(title="Train/test class distribution", height = 700)

fig.show()
def load_image(img_path):

    """

    load image from a path and preprocess it 

    """

    img = image.load_img(img_path, target_size=(224,224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    return img
def get_predictions(_model):

    """

    This function takes a trained model and gives top 3 predictions for a set of images

    """

    f, ax = plt.subplots(1,5)

    f.set_size_inches(100,20)

    for i in range(5):

        ax[i].imshow(Image.open(imgs1[i]).resize((200,200)))

    plt.show()

                  

    f, ax = plt.subplots(1,5)

    f.set_size_inches(100,20)

    for i,img_path in enumerate(imgs1):

        img = load_image(img_path)

        preds = decode_predictions(_model.predict(img), top=3)[0]

        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color='teal', ax=ax[i])

        b.tick_params(labelsize=70)

        f.tight_layout()
from keras.applications.vgg16 import VGG16

vgg16_weights = "../input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

vgg16_model = VGG16(weights=vgg16_weights)

get_predictions(vgg16_model)
from keras.applications import ResNet50

resnet50_weights = "../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5"

resnet50_model = ResNet50(weights=resnet50_weights)

get_predictions(resnet50_model)
from tensorflow.keras.applications import EfficientNetB5

efficientnetb5_weights = "../input/efficientnetb5-weights/efficientnetb5.h5"

eb5_model = EfficientNetB5(weights = efficientnetb5_weights)

get_predictions(eb5_model)
# Set image height and width

# Many popular architectures use an input size of 224 X 224

img_height, img_width = 224, 224

num_tr_imgs = img_data['train_imgs'].sum()

num_te_imgs = img_data['test_imgs'].sum()

batch_size = 16

print(f'No of traiing examples: {num_tr_imgs} \nNo. of test examples: {num_te_imgs}')
train_gen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_gen = image.ImageDataGenerator(rescale=1./255)



"""

flow_from_directory is a generator method 

that takes path to a directory & 

generates batches of augmented data 



"""

tr_generator = train_gen.flow_from_directory(train_path, target_size=(img_height, img_width), 

                                             batch_size=batch_size, class_mode = 'categorical')



te_generator = test_gen.flow_from_directory(test_path, target_size=(img_height, img_width),

                                           batch_size=batch_size, class_mode='categorical')
from keras.applications import ResNet50

base_model_resnet50 = ResNet50(weights ='imagenet', include_top = False)

x = base_model_resnet50.output

#Add a Global Average Pooling layer 

x = GlobalAveragePooling2D()(x)

#Add a DropOut layer for regularization

x = Dropout(0.3)(x)

#Add a dense layer

x = Dense(180, activation='relu')(x)

preds = Dense(131, activation='softmax')(x)



resnet50_model = Model(inputs=base_model_resnet50.input , outputs=preds)



#from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())



# train all layers of network

for layer in base_model_resnet50.layers:

    layer.trainable=False



resnet50_model.compile(optimizer=optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])



with tf.device("/device:GPU:0"):

    resnet50_pretrained_hist = resnet50_model.fit_generator(tr_generator, epochs=3, shuffle=True, verbose=1, validation_data = te_generator)
test_datagen = image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(unseen_data, target_size=(img_height, img_width), batch_size=1,

                                                 class_mode='categorical', shuffle=False)

fn = test_generator.filenames

nb_samples = len(fn)

            

data = load_files(unseen_data)

true_labels = data['target_names']

files = sorted(data['filenames'])
# Display some predictions



predicts = resnet50_model.predict_generator(test_generator,steps = nb_samples)



pred_idx = [np.argmax(predicts[i]) for i in range(20)]

pred_labels = []

for i in range(len(pred_idx)):

    for key in tr_generator.class_indices.keys():

        if tr_generator.class_indices[key] == pred_idx[i]:

            pred_labels.append(key)





fig = plt.figure(figsize=(16,12))

plt.title('True Label (Predicted Label) \n\n', fontsize=14)

plt.xticks([])

plt.yticks([])

for i in range(20):

    ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

    ax.imshow(image.load_img(files[i]))  

    ax.set_title(f"{true_labels[i]} ({pred_labels[i]})", \

                 color=("green" if true_labels[i] == pred_labels[i] else "red"))

plt.show()
from keras.applications import ResNet50

base_model_resnet50 = ResNet50(weights ='imagenet', include_top = False)

x = base_model_resnet50.output

#Add a Global Average Pooling layer 

x = GlobalAveragePooling2D()(x)

#Add a DropOut layer

x = Dropout(0.3)(x)

#Add a dense layer

x = Dense(180, activation='relu')(x)

preds = Dense(131, activation='softmax')(x)



resnet50_model2 = Model(inputs=base_model_resnet50.input , outputs=preds)



for layer in base_model_resnet50.layers:

    layer.trainable=True



resnet50_model2.compile(optimizer=optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])



with tf.device("/device:GPU:0"):

    resnet50_retrained_hist = resnet50_model2.fit_generator(tr_generator, epochs=5, shuffle=True, verbose=1, validation_data = te_generator)
# Display some predictions



predicts3 = resnet50_model2.predict_generator(test_generator,steps = nb_samples)



pred_idx = [np.argmax(predicts3[i]) for i in range(20)]

pred_labels = []

for i in range(len(pred_idx)):

    for key in tr_generator.class_indices.keys():

        if tr_generator.class_indices[key] == pred_idx[i]:

            pred_labels.append(key)





fig = plt.figure(figsize=(16,12))

plt.title('True Label (Predicted Label) \n\n', fontsize=14)

plt.xticks([])

plt.yticks([])

for i in range(20):

    ax = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])

    ax.imshow(image.load_img(files[i]))  

    ax.set_title(f"{true_labels[i]} ({pred_labels[i]})", \

                 color=("green" if true_labels[i] == pred_labels[i] else "red"))

plt.show()
plt.plot(resnet50_retrained_hist.history['accuracy'])

plt.plot(resnet50_retrained_hist.history['loss'])

plt.title('Train Accuracy & Loss')

plt.xlabel('epoch')

plt.xlim([0.9, 5.1])

plt.legend(['Accuracy', 'Loss'], loc='best')

plt.show()



plt.plot(resnet50_retrained_hist.history['val_accuracy'])

plt.plot(resnet50_retrained_hist.history['val_loss'])

plt.title('Val Accuracy & Loss')

plt.xlabel('epoch')

plt.xlim([0.9, 5.1])

plt.legend(['Accuracy', 'Loss'], loc='best')

plt.show()