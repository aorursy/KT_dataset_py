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
import tensorflow.keras as keras
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df
x = df.iloc[:,0:8].values
y = df.iloc[:,8].values
y
x
# Defining the Keras model
model = tf.keras.Sequential()

model.add(keras.layers.Dense(30,input_dim =8,activation ='relu' ))
model.add(keras.layers.Dense(50,activation ='relu' ))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(18,activation ='relu' ))
model.add(keras.layers.Dense(8,activation ='relu' ))
model.add(keras.layers.Dense(1,activation ='sigmoid' ))
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics= ['accuracy'])
model.fit(x,y,epochs=500,batch_size=100)
predictions = model.predict_classes(x)
predictions
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y,predictions),'\n')
print(classification_report(y,predictions))
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y,predictions)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)