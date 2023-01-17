import warnings

warnings.filterwarnings('ignore')



import os

os.listdir("../input/")
# from keras.applications.resnet50 import ResNet50

# from keras.applications.resnet50 import preprocess_input, decode_predictions

# from keras.preprocessing import image

# import numpy as np



# model = ResNet50(weights='imagenet')



# elephant_photo_path = "../input/elephant1.jpeg"

# beach_photo_path = "../input/beach-photo/IMG_3099.JPG"



# img = image.load_img(beach_photo_path, target_size = (224,224))

# x = image.img_to_array(img)

# x = np.expand_dims(x, axis=0)

# x = preprocess_input(x)



# preds = model.predict(x)



# # decode the results into a list of tuples (class, description, probability)

# # (one such list for each sample in the batch)



# print("predictions: ", decode_predictions(preds, top = 3)[0])

# from keras.applications.vgg16 import VGG16

# from keras.applications.vgg16 import preprocess_input, decode_predictions

# from keras.preprocessing import image

# import numpy as np



# model = VGG16(weights='imagenet', include_top = False)



# elephant_img_path = "../input/elephant-photo-for-image-classification/elephant1.jpeg"

# img = image.load_img(elephant_img_path, target_size=(224,224))

# x = image.img_to_array(img)

# x = np.expand_dims(x, axis=0)

# x = preprocess_input(x)



# features = model.predict(x)

# print(features[0][0][0])
from keras.applications.vgg19 import VGG19

from keras.applications.vgg19 import preprocess_input

from keras.preprocessing import image

from keras.models import Model

import numpy as np



base_model = VGG19(weights='imagenet')

model = Model(inputs = base_model.input, outputs = base_model.get_layer('block4_pool').output)



elephant_img_path = '../input/elephant-photo-for-image-classification/elephant1.jpeg'

img = image.load_img(elephant_img_path, target_size=(224,224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



block4_pool_features = model.predict(x)



#print(block4_pool_features[0][0][0])