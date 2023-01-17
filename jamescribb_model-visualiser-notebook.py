import os

import random

import numpy as np

import pandas as pd

import tensorflow as tf



import cv2

import matplotlib.pyplot as plt



from efficientnet import *



import keras.backend as K

from keras import layers, models

from keras.applications import DenseNet121, MobileNetV2

from keras.callbacks import Callback, ModelCheckpoint

from keras.models import Model, load_model, Sequential

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split   

from keras.preprocessing.image import apply_channel_shift

from scipy.ndimage.filters import convolve, gaussian_filter, sobel
K.clear_session()



# Weights for each model

MOBILENETV2_WEIGHTS = "../input/mobilenetv2-2015-2019-messidor-full-cropped/run9.h5"

EFFICIENTNETB0_WEIGHTS = "../input/effcientnetb0-weights/weights.h5"

EFFICIENTNETB1_WEIGHTS = "../input/efficentnetb1-weights/weights.h5"



# Layers to get visualisation information from

MOBILENETV2_LAYER = 'out_relu'

EFFICIENTNETB0_LAYER = 'swish_49'

EFFICIENTNETB1_LAYER = "swish_118"



PATH_TO_IMAGE_ARRAY_224 = "../input/image-resizer-224-224-crop/2019_Cropped.npy"

PATH_TO_IMAGE_ARRAY_240 = "../input/image-resizer-240-240-crop/2019_Cropped.npy"

PATH_TO_CSV = "../input/aptos2019-blindness-detection/train.csv"



df = pd.read_csv(PATH_TO_CSV)

images_224 = np.load(PATH_TO_IMAGE_ARRAY_224)

images_240 = np.load(PATH_TO_IMAGE_ARRAY_240)



# Backbones for each model

effNetB0 = EfficientNetB0(

    weights = None,

    include_top = False,

    input_shape = (None, None, 3)

)



effNetB1 = EfficientNetB1(

    weights = None,

    include_top = False,

    input_shape = (None, None, 3)

)



mobilenet = MobileNetV2(

    weights = None,

    include_top = False,

    input_shape = (None,None, 3)

)



class ModelInfo():

    def __init__(self, backbone, weights, last_layer, images):

        self.backbone = backbone

        self.weights = weights

        self.last_layer = last_layer

        self.images = images



mobileNetInfo = ModelInfo(mobilenet, MOBILENETV2_WEIGHTS, MOBILENETV2_LAYER, images_224)

efficientNetB0Info = ModelInfo(effNetB0, EFFICIENTNETB0_WEIGHTS, EFFICIENTNETB0_LAYER, images_224)

efficientNetB1Info = ModelInfo(effNetB1, EFFICIENTNETB1_WEIGHTS, EFFICIENTNETB1_LAYER, images_240)



# Only need to change this line

currentModel = mobileNetInfo



def build_visualisation_model(backbone, weights):

    

#     K.clear_session()

    

    GAP_layer = layers.GlobalAveragePooling2D()

    drop_layer = layers.Dropout(0.5)

    dense_layer = layers.Dense(4, activation='sigmoid')



    model = Sequential()

    model.add(backbone)

    model.add(GAP_layer)

    model.add(drop_layer)

    model.add(dense_layer)

    

    model.load_weights(weights)



    base_model = backbone

    x = GAP_layer(base_model.layers[-1].output)

    x = drop_layer(x)

    final_output = dense_layer(x)

    model = Model(base_model.layers[0].input, final_output)

    return model

model = build_visualisation_model(

    currentModel.backbone,

    currentModel.weights

)



print(model.summary())
"""

img - the image to visualise (preprocessed)

model0 - the functional model

layer_name - the name of the last conv layer [out_relu, I think]

viz_img - the image that the heatmap will be overlaid on

    (May wish to use a preprocessed image for this - seems to show up better)      

"""

def gen_heatmap_img(img, model0, layer_name='last_conv_layer',viz_img=None,orig_img=None,

    img_position=-1):

    

    # Promote the image to an array of length 1, then pass the array to the model to predict

    preds_raw = model0.predict(img[np.newaxis])

    # Convert the prediction to categorical form

    preds = preds_raw > 0.5 

    class_idx = (preds.astype(int).sum(axis=1) - 1)[0]



    # Because we are using multilabel encoding, the output tensor will consist of multiple 

    # nodes. Note that prediction = 0 implies class_idx = -1 = 3. So the whole output layer

    # will be examined for 0-level images. 

    class_output_tensor = model0.output[:, class_idx]

    

    viz_layer = model0.get_layer(layer_name)

    # Returns the derivatives of class_output_tensor with respect to viz_layer.output

    # Essentially, identifies the extent to which each image region at the final convolution

    # layer contributed to the final classification. 

    grads = K.gradients(

                        class_output_tensor ,

                        viz_layer.output

                        )[0] 

    

    # Average the gradients by image region (there are multiple convolution matrices for

    # each region)

    pooled_grads=K.mean(grads,axis=(0,1,2))



    # Return the pooled gradients and raw output for each pixel in the input image (?)

    iterate=K.function([model0.input],[pooled_grads, viz_layer.output[0]]) 

    pooled_grad_value, viz_layer_out_value = iterate([img[np.newaxis]])



    # Multiply each output value at the final convolution layer by the extent to which

    # it contributed to the final classification

    for i in range(pooled_grad_value.shape[0]):

        viz_layer_out_value[:,:,i] *= pooled_grad_value[i]

    # Average the result for each region of the image

    heatmap = np.mean(viz_layer_out_value, axis=-1)

    # Clamp all negative values to 0

    heatmap = np.maximum(heatmap,0)

    # Normalise the heatmap

    heatmap /= np.max(heatmap)



    # Standardise size of visualisation image and heatmap

    viz_img=cv2.resize(viz_img,(img.shape[1],img.shape[0]))

    heatmap=cv2.resize(heatmap,(viz_img.shape[1],viz_img.shape[0]))

    # Apply preset colour map from OpenCV to the heatmap

    # For COLORMAP_JET, low = red --> high = blue

    heatmap_color = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)/255

    # Overlay the visualisation image and the heatmap, with different degrees of transparency

    heated_img = heatmap_color*0.3 + viz_img*0.5

    

    display_images(img, viz_img, heatmap_color, heated_img, preds_raw, img_position)

    

    plt.show()

    return heated_img



# Source: https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

def gaussian_preprocess(img):

    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)

    return img/255



def display_images(img, viz_img, heatmap_color, heated_img, preds_raw, img_position):

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(10, 4))

    

    image_id = "Not found" if img_position == -1 else df.iat[img_position, 0]



    preds_raw = preds_raw[0, :]

    preds = []

    for i, pred in enumerate(preds_raw):

        preds.append('%.3f' % preds_raw[i])

    preds = '[%s]' % ', '.join(map(str, preds))



    predicted = sum(preds_raw > 0.5)



    actual = "NA" if 'diagnosis' not in df.columns else df.iat[img_position, 1] 



    fig.suptitle(f"Image: {image_id} \n\n Output: {preds} \n\n Predicted: {predicted}     Actual: {actual}")



    for axis in axs:

        axis.get_xaxis().set_visible(False)

        axis.get_yaxis().set_visible(False)

    

    axs[0].imshow(img)

    axs[0].set_title("Original")



    axs[1].imshow(viz_img)

    axs[1].set_title("Processed")



    axs[2].imshow(heatmap_color)

    axs[2].set_title("Heatmap")



    axs[3].imshow(heated_img)

    axs[3].set_title("Overlay")
for i in range(5):

    gen_heatmap_img(

    currentModel.images[i, :, :, :], 

    model, 

    layer_name=currentModel.last_layer, 

    viz_img=gaussian_preprocess(currentModel.images[i, :, :, :]),

    img_position=i

    )
diagnosis = 2

num_to_display = 5



for i in range(len(currentModel.images)):

    if num_to_display == 0: break

    if df.iat[i, 1] == diagnosis:

        num_to_display -= 1

        gen_heatmap_img(

            currentModel.images[i, :, :, :], 

            model, 

            layer_name=currentModel.last_layer, 

            viz_img=gaussian_preprocess(currentModel.images[i, :, :, :]),

            img_position=i

        )

        

        
# 0 = correct prediction; 4 = was 0, predicted 4 (or vice versa)

degree_of_error = 4

num_to_display = 5



for i in range(len(currentModel.images)):

    if num_to_display == 0: break

    img = currentModel.images[i, :, :, :]

    img = np.expand_dims(img, axis=0)

    y_pred = (model.predict(img) > 0.5).sum(axis=1)

    if abs(y_pred - df.iat[i, 1]) == degree_of_error: 

        num_to_display -= 1

        gen_heatmap_img(

            currentModel.images[i, :, :, :], 

            model, 

            layer_name=currentModel.last_layer, 

            viz_img=gaussian_preprocess(currentModel.images[i, :, :, :]),

            img_position=i

        )
num_to_display = 5



for i in range(num_to_display):

    if num_to_display == 0: break

    n = random.randrange(len(currentModel.images))

#     img = currentModel.images[n, :, :, :]

#     img = np.expand_dims(img, axis=0)

    num_to_display -= 1

    gen_heatmap_img(

        currentModel.images[n, :, :, :], 

        model, 

        layer_name=currentModel.last_layer, 

        viz_img=gaussian_preprocess(currentModel.images[n, :, :, :]),

        img_position=n

    )