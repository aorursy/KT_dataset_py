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
tf.test.gpu_device_name()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
import cv2
import os
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3()
model.summary()
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions
chair = []

for filepath in tqdm(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/chair/')):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/chair/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    chair.append(src)

chair = np.array(chair)

bed = []

for filepath in tqdm(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/bed/')):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    bed.append(src)

bed = np.array(bed)

table = []

for filepath in tqdm(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/')):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/table/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    table.append(src)

table = np.array(table)

sofa = []

for filepath in tqdm(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/sofa/')):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/sofa/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    sofa.append(src)

sofa = np.array(sofa)

swivle_chair = []

for filepath in tqdm(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/swivelchair/')):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/train/swivelchair/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    swivle_chair.append(src)

swivle_chair = np.array(swivle_chair)


print(chair.shape, table.shape, bed.shape, swivle_chair.shape, sofa.shape)
image = chair[0]

plt.imshow(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

model = InceptionV3()

yhat = model.predict(image)

label = decode_predictions(yhat)

label = label[0][0]
print('%s (%.5f%%)' % (label[1], label[2]*100))
print(yhat.shape, '\n', yhat)
x = np.array([i for i in chair] + [i for i in table] + [i for i in bed] + [i for i in swivle_chair] + [i for i in sofa])
y_sparse = np.array([1 for _ in range(chair.shape[0])] + [4 for _ in range(table.shape[0])] + [0 for _ in range(bed.shape[0])] + [3 for _ in range(swivle_chair.shape[0])] + [2 for _ in range(sofa.shape[0])])
y_encoded = []

for i in y_sparse:
    a = [0 for _ in range(5)]
    a[i] = 1
    y_encoded.append(a)
y_encoded = np.array(y_encoded)

print(x.shape, y_encoded.shape)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, Concatenate, Dropout
import tensorflow as tf
model = InceptionV3(include_top=False, input_shape=(256, 256, 3))
model.trainable = True

gap1 = GlobalAveragePooling2D()(model.layers[-1].output)
flat1 = Flatten()(gap1)
class1 = Dense(1024, activation='relu')(flat1)
class1 = Dropout(0.1)(class1)
class2 = Dense(256, activation='relu')(class1)
class2 = Dropout(0.1)(class2)
class3 = Dense(32, activation='relu')(class2)
output = Dense(5, activation='softmax')(class3)

model = Model(inputs=model.inputs, outputs=output)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x, y_encoded, epochs=5, validation_split=0.1, shuffle=True)
test = []
files = sorted(os.listdir('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/'))
for filepath in tqdm(files):
    src = cv2.imread('/kaggle/input/day-3-kaggle-competition/data_comp/data_comp/test/{0}'.format(filepath),1)
    src = cv2.resize(src, (256, 256))
    test.append(src)

test = np.array(test)
y_pred = model.predict(test, verbose=1)
y_peed = [np.argmax(i) for i in y_pred]
submission = pd.DataFrame({'image': files, 'target':y_peed})
submission.to_csv('submission_basic.csv', index=False)
model1 = InceptionV3(include_top=False, input_shape=(256, 256, 3))
model1.trainable = True

gap1 = GlobalAveragePooling2D()(model1.layers[-1].output)
flat1 = Flatten()(gap1)
out1 = Dense(1024, activation='relu')(flat1)

input2 = tf.keras.layers.Input([256, 256, 3])
x = tf.keras.applications.inception_v3.preprocess_input(input2)
model2 = tf.keras.applications.InceptionV3()
out2 = model2(x)

mergedOut = Concatenate()([out1,out2])
class1 = Dense(1024, activation='relu')(mergedOut)
class1 = Dropout(0.1)(class1)
class1 = Dense(1024, activation='relu')(class1)
class1 = Dropout(0.1)(class1)
class1 = Dense(516, activation='relu')(class1)
class1 = Dropout(0.1)(class1)
class2 = Dense(256, activation='relu')(class1)
class2 = Dropout(0.1)(class2)
class3 = Dense(32, activation='relu')(class2)
output = Dense(5, activation='softmax')(class3)


model = Model(inputs=[model1.inputs, input2], outputs=[output])
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
print(x.shape, y_encoded.shape)
model.fit([x, x], y_encoded, epochs=250, shuffle=True, batch_size=32, validation_split=0.15)
