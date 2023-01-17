%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
!ls ../input
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
!cp ../input/resnet* ~/.keras/models/
!cp ../input/imagenet_class_index.json ~/.keras/models/
!ls ~/.keras/models
fig, ax = plt.subplots(1, figsize=(12, 10))
img = image.load_img('../input/Kuszma.JPG')
img = image.img_to_array(img)
ax.imshow(img / 255.) 
ax.axis('off')
plt.show()
resnet = ResNet50(weights='imagenet')
img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
print(img.max(),img.min())
plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
print(x.max(),x.min())
preds = resnet.predict(x)
decode_predictions(preds, top=5)
img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
print(img.max(),img.min())
plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0),mode='tf')
print(x.max(),x.min())
preds = resnet.predict(x)
decode_predictions(preds, top=5)
img = image.load_img('../input/Kuszma.JPG', target_size=(224, 224))
img = image.img_to_array(img)
print(img.max(),img.min())
plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0),mode='torch')
print(x.max(),x.min())
preds = resnet.predict(x)
decode_predictions(preds, top=5)