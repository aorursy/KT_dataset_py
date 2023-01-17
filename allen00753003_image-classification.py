import os

print(os.listdir('../input/000000'))
from PIL import Image

import matplotlib.pyplot as plt
img = Image.open('../input/000000/naruto.jpg')

img = img.resize((224,224))

plt.imshow(img)
import numpy as np

test_x = np.array(img) / 255.0

print(test_x.shape)
test_x = test_x.reshape(1,224,224,3)
# Import Model

#from tensorflow.keras.applications import VGG16

#from tensorflow.keras.applications import ResNet101V2

from tensorflow.keras.applications import InceptionV3



#from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

#from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions

from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions



# Load Model

#model = VGG16(weights='imagenet')

#model = ResNet101V2(weights='imagenet')

model = InceptionV3(weights='imagenet')
# model prediction

preds = model.predict(test_x)

# decode prediction

dec_preds =  decode_predictions(preds, top=3)[0]

print('Predicted:', dec_preds)