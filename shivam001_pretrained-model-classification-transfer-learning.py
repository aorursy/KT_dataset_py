# By using ResNet50 pretrained model
# It's called transfer learning
import os
print(os.listdir('../input'))

from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model_resnet = ResNet50()
img_path_ = '../input/Kuszma.JPG'
img = load_img(img_path_, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model_resnet.predict(x)
# top 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])

import matplotlib.pyplot as plt
img=plt.imread('../input/Kuszma.JPG')
plt.imshow(img)
plt.grid(False)
label_resnet=decode_predictions(preds)
label_resnet=label_resnet[0][0]
plt.title('%s(%.2f%%)'%(label_resnet[1],label_resnet[2]*100))
plt.show()
'''
decode_predictions use to convert probability to class_label
preprocess_input use to make image for vgg16
'''
# By using VGG!16 pretrained model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras.models import Model
import numpy as np

base_model = VGG16()
img_path = '../input/Kuszma.JPG'
img_1 =load_img(img_path, target_size=(224, 224))
img_1 =img_to_array(img_1)
x_=img_1.reshape((1,img_1.shape[0],img_1.shape[1],img_1.shape[2]))

x_ = preprocess_input(x_)
prediction = base_model.predict(x_)
class_label = decode_predictions(prediction)
class_label = class_label[0][0]
print('%s (%.2f%% probability)' % (class_label[1], class_label[2]*100))

import matplotlib.pyplot as plt
img_show=plt.imread('../input/Kuszma.JPG')
plt.imshow(img_show)
plt.grid(False)
plt.title('%s (%.2f%% probability)' % (class_label[1], class_label[2]*100))
plt.show()

