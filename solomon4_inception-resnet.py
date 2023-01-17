import numpy as np

import tensorflow.keras  

from tensorflow.keras.preprocessing.image import img_to_array,load_img

from IPython.display import Image

import tensorflow as tf

from tensorflow.keras import backend as k
Image(filename='/kaggle/input/natural-images/natural_images/fruit/fruit_0862.jpg')
a = load_img('/kaggle/input/natural-images/natural_images/fruit/fruit_0862.jpg', target_size=(299,299))

image = img_to_array(a)

paddedimage = np.expand_dims(image, axis=0) 

 

 
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions

inception_model = InceptionResNetV2(weights='imagenet')
p = inception_model.predict(paddedimage)
decode_predictions(p)
from keras.applications.inception_resnet_v2 import preprocess_input
preprocessed = preprocess_input(paddedimage)

p = inception_model.predict(paddedimage)

decode_predictions(p)
