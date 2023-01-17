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
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization
import time

start_time = time.time()

train_datagen = ImageDataGenerator(rescale=1./255,
								   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',
	                                              batch_size=32,
                                                  target_size=(150, 150),
                                                  class_mode='categorical')

test_set = test_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode='categorical')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform', 
	                    input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(400, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(128, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(6, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_set, epochs = 30)

preds = model.evaluate(test_set)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.summary()
n = 0
dic = {0 : 'Buildings',
       1 : 'Forest',
       2 : 'Glacier',
       3 : 'Mountain',
       4 : 'Sea',
       5 : 'Street'}

image_name = list()
label = list()
image_type = list()
print('Predicting Images......')
for img in range(24333) :
	try :
		img_path = '../input/intel-image-classification/seg_pred/seg_pred/' + str(img) + '.jpg'
		test_image = image.load_img(img_path, target_size = (150, 150))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		test_image = test_image/255.0
		result = model.predict(test_image)
		q = np.argmax(result)
		n += 1
		if n%200 == 0 :
			print('Images Predicted',n)
		image_name.append(str(img))
		label.append(q)
		z = dic.get(q)
		image_type.append(z)
	except :
		continue

list_of_tuples = list(zip(image_name, label, image_type))
print('Total Images Predicted',n)  

df2 = pd.DataFrame(list_of_tuples, columns = ['Image_Name', 'Image_label','Image_Type'])
df2.to_csv('Predicted_classes.csv', index = False)