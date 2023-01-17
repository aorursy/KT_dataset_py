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
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import tensorflow as tf
train=pd.read_csv('../input/digit-recognizer/train.csv')
test=pd.read_csv('../input/digit-recognizer/test.csv')
train.head()

test.head()
y_train=train['label']
X_train=train.drop('label',axis=1)
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
X_train=X_train/255
y_train=tf.keras.utils.to_categorical(y_train)
plt.imshow(X[65][:,:,0])
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=.1)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
    
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(patience=2,monitor='loss')
model.fit(x=X_train,y=y_train,validation_data=(X_val,y_val),epochs=20,callbacks=[early_stop])
prd=model.predict_classes(test)
prd_table=pd.DataFrame(prd)
test_to_check=pd.read_csv('../input/digit-recognizer/test.csv')
test_to_check.head()
submit=pd.concat([test_to_check,prd_table],axis=1)
final=pd.DataFrame(submit[0])
final=final.reset_index()
final.columns=['ImageId','Label']
def a(x):
    return x+1
final['ImageId']=final['ImageId'].apply(a)
final
final.to_csv('mycsvfile.csv',index=False)

