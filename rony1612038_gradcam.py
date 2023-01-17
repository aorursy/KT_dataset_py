import os





import copy

import warnings

warnings.filterwarnings('ignore')



import cv2

import keras

from keras import backend as K

from keras.models import Model, Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input

from keras.layers import Conv2D, Activation, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img, img_to_array

from keras.applications.resnet50 import preprocess_input, ResNet50

import matplotlib

import matplotlib.pylab as plt

import numpy as np

import seaborn as sns

import shap

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
W = 224 # The default size for ResNet is 224 but resize to .5 to save memory size

H = 224 # The default size for ResNet is 224 but resize to .5 to save memory size

label_to_class = {

    'covid': 0,

    'normal':    1,

    'pneumonia':   2

}

class_to_label = {v: k for k, v in label_to_class.items()}

n_classes = len(label_to_class)



def get_images(dir_name='../input/preprocessed-ekdom-latest/preprocessed', label_to_class=label_to_class):

    """read images / labels from directory"""

    

    Images = []

    Classes = []

    

    for label_name in os.listdir(dir_name):

        cls = label_to_class[label_name]

        

        for img_name in os.listdir('/'.join([dir_name, label_name])):

            img = load_img('/'.join([dir_name, label_name, img_name]), target_size=(W, H))

            img = img_to_array(img)

            

            Images.append(img)

            Classes.append(cls)

            

    Images = np.array(Images, dtype=np.float32)

    Classes = np.array(Classes, dtype=np.float32)

    Images, Classes = shuffle(Images, Classes, random_state=0)

    

    return Images, Classes
## get images / labels



Images, Classes = get_images()



Images.shape, Classes.shape
## visualize some images / labels



n_total_images = Images.shape[0]



for target_cls in [0, 1, 2]:

    

    indices = np.where(Classes == target_cls)[0] # get target class indices on Images / Classes

    n_target_cls = indices.shape[0]

    label = class_to_label[target_cls]

    print(label, n_target_cls, n_target_cls/n_total_images)



    n_cols = 10 # # of sample plot

    fig, axs = plt.subplots(ncols=n_cols, figsize=(25, 3))



    for i in range(n_cols):



        axs[i].imshow(np.uint8(Images[indices[i]]))

        axs[i].axis('off')

        axs[i].set_title(label)



    plt.show()
## split train / test



indices_train, indices_test = train_test_split(list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=False)



x_train = Images[indices_train]

y_train = Classes[indices_train]

x_test = Images[indices_test]

y_test = Classes[indices_test]



x_train.shape, y_train.shape, x_test.shape, y_test.shape
## to one-hot



y_train = keras.utils.to_categorical(y_train, n_classes)

y_test = keras.utils.to_categorical(y_test, n_classes)



y_train.shape, y_test.shape
## to image data generator



datagen_train = ImageDataGenerator(

    preprocessing_function=preprocess_input, # image preprocessing function

    rotation_range=30,                       # randomly rotate images in the range

    zoom_range=0.1,                          # Randomly zoom image

    width_shift_range=0.1,                   # randomly shift images horizontally

    height_shift_range=0.1,                  # randomly shift images vertically

    horizontal_flip=True,                    # randomly flip images horizontally

    vertical_flip=False,                     # randomly flip images vertically

)

datagen_test = ImageDataGenerator(

    preprocessing_function=preprocess_input, # image preprocessing function

)
import keras

import tensorflow as tf

from keras.applications.mobilenet_v2 import MobileNetV2



def build_model():

    """build model function"""

    

    # Resnet

    input_tensor = Input(shape=(W, H, 3)) # To change input shape

    densenet_121 = MobileNetV2(

        include_top=False,                # To change output shape

        weights='imagenet',               # Use pre-trained model

        input_tensor=input_tensor,        # Change input shape for this task

    )

    

    # fc layer

    top_model = Sequential()

    top_model.add(GlobalAveragePooling2D())               # Add GAP for cam

    top_model.add(Dense(n_classes, activation='softmax')) # Change output shape for this task

    

    # model

    model = Model(input=densenet_121.input, output=top_model(densenet_121.output))

    

    # frozen weights

    for layer in model.layers[:-11]:

        layer.trainable = False or isinstance(layer, BatchNormalization) # If Batch Normalization layer, it should be trainable

        

    # compile

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
model = build_model()
model.summary()
## finetuning



batch_size = 32



history = model.fit_generator(

    datagen_train.flow(x_train, y_train, batch_size=32),

    epochs= 50,

    validation_data=datagen_test.flow(x_test, y_test, batch_size=32),

    verbose = 1,

    #callbacks=callbacks,

    steps_per_epoch=  int(len(x_train)//batch_size),

    validation_steps= int(len(x_test)// batch_size)

)
## plot confusion matrix



x = preprocess_input(copy.deepcopy(x_test))

y_preds = model.predict(x)

y_preds = np.argmax(y_preds, axis=1)

y_trues = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_trues, y_preds)



fig, ax = plt.subplots(figsize=(7, 6))



sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)



ax.set(

    xticklabels=list(label_to_class.keys()),

    yticklabels=list(label_to_class.keys()),

    title='confusion matrix',

    ylabel='True label',

    xlabel='Predicted label'

)

params = dict(rotation=45, ha='center', rotation_mode='anchor')

plt.setp(ax.get_yticklabels(), **params)

plt.setp(ax.get_xticklabels(), **params)

plt.show()
def superimpose(img, cam):

    """superimpose original image and cam heatmap"""

    

    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)



    superimposed_img = heatmap * .45 + img * 1.2

    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  

    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    

    return img, heatmap, superimposed_img
def _plot(model, cam_func, img, cls_true):

    """plot original image, heatmap from cam and superimpose image"""

    

    # for cam

    x = np.expand_dims(img, axis=0)

    x = preprocess_input(copy.deepcopy(x))



    # for superimpose

    img = np.uint8(img)



    # cam / superimpose

    cls_pred, cam = cam_func(model=model, x=x, layer_name=model.layers[-2].name)

    img, heatmap, superimposed_img = superimpose(img, cam)



    fig, axs = plt.subplots(ncols=2, figsize=(8, 6))



    axs[0].imshow(img)

    #axs[0].set_title('True label: ' + class_to_label[cls_true] + ' / Predicted label : ' + class_to_label[cls_pred])

    axs[0].axis('off')



    #axs[1].imshow(heatmap)

    #axs[1].set_title('heatmap')

    #axs[1].axis('off')



    axs[1].imshow(superimposed_img)

    #axs[1].set_title(class_to_label[cls_true])

    axs[1].axis('off')



    plt.suptitle('True label: ' + class_to_label[cls_true] + ' / Predicted label : ' + class_to_label[cls_pred])

    plt.tight_layout()

    plt.show()

    #fig.savefig("colon_aca_prewitt.jpeg",bbox_inches='tight', pad_inches=0)

    

    
## Grad-CAM function



def grad_cam(model, x, layer_name):

    """Grad-CAM function"""

    

    cls = np.argmax(model.predict(x))

    

    y_c = model.output[0, cls]

    conv_output = model.get_layer(layer_name).output

    grads = K.gradients(y_c, conv_output)[0]



    # Get outputs and grads

    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([x])

    output, grads_val = output[0, :], grads_val[0, :, :, :]

    

    weights = np.mean(grads_val, axis=(0, 1)) # Passing through GlobalAveragePooling



    cam = np.dot(output, weights) # multiply

    cam = np.maximum(cam, 0)      # Passing through ReLU

    cam /= np.max(cam)            # scale 0 to 1.0



    return cls, cam
for i in range(200):



    _plot(model=model, cam_func=grad_cam, img=Images[i], cls_true=Classes[i])
