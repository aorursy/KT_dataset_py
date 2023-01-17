# Check docker environment

import os

os.getcwd()
os.listdir()
os.listdir('../')
os.listdir('../input/')
os.listdir('../input/test')
os.listdir('../input/train/')
len(os.listdir('../input/train/')), len(os.listdir('../input/test/'))
os.listdir('../lib')
os.listdir('../lib/kagglegym/')
os.listdir('../config/')
from keras.preprocessing import image
files = os.listdir('../input/train')
for j,f in enumerate(files):

    img = image.load_img('../input/train/'+f)

    print(j,f,img.height,img.width)
img = image.load_img('../input/train/'+files[11])
import matplotlib.pyplot as plt

plt.imshow(img)
x = image.img_to_array(img)
x.min(),x.max()
x.dtype
x = (x-x.min())/(x.max()-x.min())
x.dtype,x.min(),x.max()
sample_dog = image.load_img('../input/train/dog.11987.jpg')
plt.imshow(sample_dog)
sample_cat = image.load_img('../input/train/cat.12253.jpg')

plt.imshow(sample_cat)
sample_dog.height,sample_dog.width,sample_cat.height,sample_cat.width
import numpy as np
np.array(sample_dog).shape,np.array(sample_cat).shape
# With resizing

sample_dog = image.load_img('../input/train/dog.10041.jpg',target_size=(150,150))

plt.imshow(sample_dog)
sample_cat = image.load_img('../input/train/cat.9265.jpg',target_size=(150,150))

plt.imshow(sample_cat)
from keras.applications.xception import Xception

from keras.applications.xception import preprocess_input

from keras.applications.imagenet_utils import decode_predictions

from keras.preprocessing import image



model = Xception()





def pred (m,n):

	for j in range(m,n):

		file = files[j]

		img = image.load_img('../input/train/'+file,target_size=(299,299))

		x = image.img_to_array(img)

		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		preds = model.predict(x)

		print (decode_predictions(preds)[0][0][1], ', ', file.split('.')[0])
# Cannot download pretrained model

len(files)
import keras
# Build from scratch

model = keras.models.Sequential()
model.add()