# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
data_dir='../input/cell-images-for-detecting-malaria'
print(os.listdir(data_dir))
print(os.listdir(data_dir+'/cell_images'))
print(os.listdir(data_dir+'/cell_images'+'/cell_images'))
print(os.listdir(data_dir+'/cell_images'+'/cell_images'+'/Uninfected'))
print(os.listdir(data_dir+'/cell_images'+'/cell_images'+'/Parasitized'))
from matplotlib.image import imread
uninfected=data_dir+'/cell_images'+'/cell_images'+'/Uninfected'+'/C130P91ThinF_IMG_20151004_142951_cell_89.png'
parasitized=data_dir+'/cell_images'+'/cell_images'+'/Parasitized'+'/C186P147NThinF_IMG_20151203_150408_cell_170.png'
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.imshow(imread(uninfected))
plt.title('uninfected_image')
plt.xticks([]) , plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(imread(parasitized))
plt.title('infected_image')
plt.xticks([]) , plt.yticks([])
imread(uninfected).shape
imread(parasitized).shape
print('no of uninfected data:',len(os.listdir(data_dir+'/cell_images'+'/cell_images'+'/Uninfected')))
print('no of parasitized data:',len(os.listdir(data_dir+'/cell_images'+'/cell_images'+'/Parasitized')))
width=[]
height=[]

for image_name in os.listdir(data_dir+'/cell_images'+'/cell_images'+'/Uninfected'):
    img=imread(data_dir+'/cell_images'+'/cell_images'+'/Uninfected/'+image_name)
    d1,d2,color=img.shape
    width.append(d1)
    height.append(d2)

width
height
sns.jointplot(width,height)
np.mean(width)
np.mean(height)
# we can fix our height and width
fix_width=130
fix_height=130
# there is no need to rescale our data as they are already between 0 and 1
print(imread(uninfected).max())
print(imread(parasitized).max())
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen=ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            validation_split=0.2)
train=image_gen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                   target_size=(fix_width,fix_height),
                                    color_mode='rgb',
                                    class_mode = 'binary',
                                    batch_size = 16,
                                    subset='training')
test=image_gen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                   target_size=(fix_width,fix_height),
                                   color_mode='rgb',
                                    class_mode = 'binary',
                                    batch_size = 16,
                                   shuffle=False,
                                    subset='validation')
train.class_indices
test.class_indices
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(130,130,3)))
model.add(MaxPool2D(2,2))


model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(130,130,3)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(130,130,3)))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=2)
results=model.fit_generator(train,
                           epochs=20,
                           validation_data=test,
                           callbacks=[early_stop])
predictions=model.predict_generator(test)
pred=predictions>0.5
pred
from sklearn.metrics import classification_report
print(classification_report(pred,test.classes))
