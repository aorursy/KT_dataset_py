from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#import required libraries
%matplotlib inline

from os import makedirs
from os.path import join, exists, expanduser
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.preprocessing import image
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.utils.data_utils import GeneratorEnqueuer
import math,os
import glob
import time
%matplotlib inline
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

!ls ../input/keras-pretrained-models/
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
!cp ../input/keras-pretrained-models/inception_resnet_v2* ~/.keras/models/
!ls ~/.keras/models
model = InceptionResNetV2(weights='imagenet')
print("Inception ResNet V2 model loaded")
#model to load each image, pre-process it and predict
def model_predict(images):
    #predictions.clear()
    for i in range(len(images)):
        images[i] = image.load_img(images[i], target_size=(224, 224))
        imagecopy = images[i].copy()
        images[i] = image.img_to_array(images[i])
        x = preprocess_input(np.expand_dims(images[i].copy(), axis=0))
        preds = model.predict(x)
        predictions = (decode_predictions(preds, top=1))
        
        #Display the picture and its predicted label
        display_pipeline(imagecopy,predictions)
    return predictions
def display_pipeline(imgcpy,predictions):
    plt.imshow(imgcpy)
    plt.show()
#     print((predictions[2]*100),"% probability that this is a ",predictions[1])
#     print("imageid/str:", predictions[0])
    print(predictions)
#Import image directory files
images = glob.glob('../input/google-ai-open-images-object-detection-track/test/challenge2018_test/*')
images_new = images[25:27]
print("Import done!")
#Call predict function, calculate time to predict
#predictions = []
start = time.time()
predictions = model_predict(images_new)
end = time.time()
print("Total time taken for object detection on",len(images_new),"images: ",round((end-start),2),"seconds")

