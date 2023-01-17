import numpy as np
from PIL import Image
from keras.applications import *
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
import os
flowers = os.listdir('../input/flowers-recognition/flowers/flowers')
print (flowers)
paths = {'rose': [],
         'dandelion': [],
         'sunflower': [],
         'tulip': [],
         'daisy': []
        }

for key, images in paths.items():
    for filename in os.listdir('../input/flowers-recognition/flowers/flowers/'+key):
        paths[key].append('../input/flowers-recognition/flowers/flowers/'+key+'/'+filename)
    
    print (len(images),key,'images')
X = []
Y = []
mapping = {'rose': 0,
         'dandelion': 1,
         'sunflower': 2,
         'tulip': 3,
         'daisy': 4
        }
for label,image_paths in paths.items():
    for path in image_paths:
        if '.py' not in path:
            image = Image.open(path)
            image = image.resize((224,224))
            X.append(np.array(image))

            one_hot = np.array([0.,0.,0.,0.,0.])
            one_hot[mapping[label]] = 1.
            Y.append(one_hot)
aug_X = []
aug_Y = []

for image in X:
    aug_X.append(np.flip(image,1))

aug_Y = Y
X = X + aug_X
Y = Y + aug_Y
len(X)
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras import backend as K
base_model = ResNet50(weights=None, include_top=False, input_shape=(224,224,3))
base_model.load_weights('../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
for layer in base_model.layers:
    layer.trainable = False
output = base_model.output
from keras.layers import Flatten
output = Flatten()(output)
output = Dense(5, activation='softmax')(output)
model = Model(inputs=base_model.input, outputs=output)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.stack(X,axis=0),np.stack(Y,axis=0),validation_split=0.1,batch_size=8,epochs=15,verbose=1)
