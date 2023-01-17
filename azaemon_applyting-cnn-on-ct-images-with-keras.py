import numpy as np # matrix tools
import matplotlib.pyplot as plt # for basic plots
import seaborn as sns # for nicer plots
import pandas as pd
from glob import glob
import re
from skimage.io import imread

import keras
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname,"______")
    for filename in filenames:
        print(os.path.join(dirname, filename))
BASE_IMG_PATH=os.path.join('..','kaggle/input')
path= os.path.join(BASE_IMG_PATH,'overview.csv')
overview = pd.read_csv(path)
overview.head()
len(overview)
overview['Contrast'] = overview['Contrast'].map(lambda x: 1 if x else 0)
plt.figure(figsize=(10,5))
sns.distplot(overview['Age'])
g = sns.FacetGrid(overview, col="Contrast", size=8)
g = g.map(sns.distplot, "Age")
g = sns.FacetGrid(overview, hue="Contrast",size=6, legend_out=True)
g = g.map(sns.distplot, "Age").add_legend()
BASE_IMG_PATH=os.path.join('..','kaggle/input')
print(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))
all_images_list = glob(os.path.join(BASE_IMG_PATH,'tiff_images','*.tif'))
all_images_list[:5]
print(all_images_list)
imread(all_images_list[0]).shape
np.array(np.arange(81)).reshape(9,9)
np.array(np.arange(81)).reshape(9,9)[::3,::3]
np.expand_dims(imread(all_images_list[0])[::4,::4],0).shape
jimread = lambda x: np.expand_dims(imread(x)[::2,::2],0)
test_image = jimread(all_images_list[0])
plt.imshow(test_image[0])
check_contrast = re.compile(r'data\\tiff_images\\ID_([\d]+)_AGE_[\d]+_CONTRAST_([\d]+)_CT.tif')
label = []
id_list = []
for image in all_images_list:
    id_list.append(check_contrast.findall(image)[0][0])
    label.append(check_contrast.findall(image)[0][1])
label_list = pd.DataFrame(label,id_list)
label_list.head()
images = np.stack([jimread(i) for i in all_images_list],0)
len(images)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, label_list, test_size=0.1, random_state=0)
n_train, depth, width, height = X_train.shape
n_test,_,_,_ = X_test.shape
n_train,depth, width, height
input_shape = (width,height,depth)
input_shape
input_train = X_train.reshape((n_train, width,height,depth))
input_train.shape
input_train.astype('float32')
input_train = input_train / np.max(input_train)
input_train.max()
input_test = X_test.reshape(n_test, *input_shape)
input_test.astype('float32')
input_test = input_test / np.max(input_test)
output_train = keras.utils.to_categorical(y_train, 2)
output_test = keras.utils.to_categorical(y_test, 2)
output_train[5]
input_train.shape
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
batch_size = 20
epochs = 40
model2 = Sequential()
model2.add(Conv2D(50, (5, 5), activation='relu', input_shape=input_shape))
 # 32개의 4x4 Filter 를 이용하여 Convolutional Network생성
model2.add(MaxPooling2D(pool_size=(3, 3))) # 3x3 Maxpooling 
model2.add(Conv2D(30, (4, 4), activation='relu', input_shape=input_shape))
model2.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 Maxpooling 
model2.add(Flatten()) # 쭉풀어서 Fully Connected Neural Network를 만든다. 
model2.add(Dense(2, activation='softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
history = model2.fit(input_train, output_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(input_test, output_test))
score = model2.evaluate(input_test, output_test, verbose=0)
score
