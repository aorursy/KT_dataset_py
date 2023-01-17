# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import random 
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
imagepaths=[]
for root,dirs,files in os.walk("/kaggle/input/soli-data/dsp",topdown=False):
    for name in files:
        path = os.path.join(root,name)
        if path.endswith("h5"):
            imagepaths.append(path)
print("Total files : ",len(imagepaths))

random.shuffle(imagepaths)
def load_data(paths_list):
    with h5py.File(paths_list[0], 'r') as f:
        data = f['ch{}'.format(0)][()]
        if(data.shape[0]<=100):
            data=np.pad(data,((0,100-data.shape[0]),(0,0)))# paddin with 100 frames
        else:
            data=data[:100]
        y = f['label'][()]
        label=y[0]

    for hfile in paths_list:
        with h5py.File(hfile, 'r') as f:
            for channel in range(4):
                rdata = f['ch{}'.format(channel)][()]
                if(rdata.shape[0]<=100):
                    rdata=np.pad(rdata,((0,100-rdata.shape[0]),(0,0)))
                else:
                    rdata=rdata[:100]
                data=np.dstack((data,rdata))
                y = f['label'][()]
                label=np.concatenate((label,y[0]))
    data=np.swapaxes(data,1,2)
    data=np.swapaxes(data,0,1)
    return data,label

print("train data: ")
train_x,train_y=load_data(imagepaths[:500])
print(train_x.shape)
print("test data: ")

test_x,test_y=load_data(imagepaths[500:750])
print(test_x.shape)
val_x,val_y=load_data(imagepaths[750:1000])

model= keras.models.Sequential([
 keras.layers.Dense(512,activation='relu'),
 keras.layers.LSTM(512),
 keras.layers.Dense(12,activation='softmax')])
# checkpoint_cb=tf.keras.callbacks.ModelCheckpoint("/home/ee/mtech/eet192341/codes/rnn_deep/rnn_deep.h5",save_best_only=True)
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics=["sparse_categorical_accuracy"])
history = model.fit(train_x,train_y,verbose=2, epochs=50,validation_data=(val_x,val_y))

pd.DataFrame(history.history).plot(figsize=(16,10))
plt.grid(True)
plt.show()
model.evaluate(test_x,test_y)
from sklearn.metrics import confusion_matrix
y_pred=model.predict_classes(test_x)
acc=(test_y==y_pred)/len(test_y)
print("accuracy ",acc)
c=confusion_matrix(test_y,y_pred)
label_name=["pinch index","palm tilt","finger slide","pinch pinky","slow swip","fast swip","push","pull","finger rub","circle","hold","background"]

fig = plt.figure(figsize=[22,18])
import seaborn as sns
sns.heatmap(c, annot=True,annot_kws={"size": 13},xticklabels=label_name,yticklabels=label_name)
