from keras.applications.resnet50 import ResNet50



model = ResNet50(weights='imagenet')
%matplotlib inline

import matplotlib.pyplot as plt

import cv2

from skimage import io



#img_path = 'https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_slide/public/images/car-reviews/first-drives/legacy/large-2479-s-classsaloon.jpg'

img_path = 'https://photos7.motorcar.com/used-2017-ferrari-488_spider-convertible-8431-19396579-1-1024.jpg'



img = io.imread(img_path)

img =cv2.resize(img,(224, 224))

imgplot = plt.imshow(img)
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np

from keras.preprocessing import image





x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)

# (one such list for each sample in the batch)

print('Predicted:', decode_predictions(preds, top=3)[0])
model.summary()