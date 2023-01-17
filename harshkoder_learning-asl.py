import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread as mread
import numpy as np
from skimage import color
import pandas as pd
from skimage.transform import resize as reimgsize
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
import cv2 as cv
def show_image(image,title="Image",cmap_type='gray'):
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
cat,i,decat={},0,[]

for c in "_0123456789ABCEFGHJKLNPQRSTUXYZ":
    decat.append(c)
    cat[c]=i
    i+=1
print(cat)
cwd='/kaggle/input/gestures'
x,y=[],[]
for folder,_,files in os.walk(cwd):
    for file in files:
        if folder[-1]!='I':
            x.append(mread(os.path.join(folder,file)))
            y.append(cat[folder[-1]])
    print("Done",folder)
X=np.array(x)
X.resize(len(x),50,50,1)
from keras.utils import np_utils
Y = np_utils.to_categorical(y)
y[-10:-1]
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils,plot_model
model=Sequential()
model.add(Conv2D(16,(2,2),input_shape=(50,50,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
#model.add(Dense(144, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(31,activation='softmax'))
#sgd = optimizers.SGD(lr=1e-2)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X,Y,epochs=3,validation_split=0.01)

model.save('gestures.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
