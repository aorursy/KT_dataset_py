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
import tensorflow as tf
import os
import shutil
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,GlobalAveragePooling2D
test_data= pd.read_csv('../input/digit-recognizer/test.csv')
train_data= pd.read_csv('../input/digit-recognizer/train.csv')
y = train_data.iloc[:,:1]
x = train_data.iloc[:,1:]
x/=255.0
x=x.values.reshape(-1,28,28,1)
print(x.shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape = (28,28,1)),
#     tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',padding = 'same'),
    tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64,(1,1),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',padding = 'same'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512,activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128,activation = 'relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10,activation = 'softmax')
])
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.summary()
history = model.fit(x,y,epochs = 100,verbose = 1)
test_data = test_data.values.reshape(-1,28,28,1)
test_data = test_data / 255.0
submit = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submit.Label =model.predict_classes(test_data)
submit.head()
submit.to_csv('submit.csv',index=False)
