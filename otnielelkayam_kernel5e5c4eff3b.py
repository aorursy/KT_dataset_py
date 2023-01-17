import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn
plt.style.use('ggplot')
%matplotlib inline

import os
import re
import pandas as  pd
data = os.walk('test-task-data').__next__()
files = np.zeros((39102,40,110,3), dtype='uint8')
for s in data[2][:]:
    x = re.search('\d+.jpg' ,s)
    if str(x) != 'None':
        files[int(x.string[:-4]) - 1] = image.imread('%s/%s' % (data[0], x.string))
#         files.append(image.imread('%s/%s' % (data[0], x.string)))
files = np.array(files)
files = files / 255.0
files.shape
y = pd.read_csv('test-task-data/data.csv', header=None)
y = y.set_index(0)
y = y.sort_index()
y.index -= 1

# "broken" data with letters istead digits
y[1][12495] = '66987'
y[1][14096] = '54109'
y[1][14191] = '84824'
y[1][16796] = '20741'
y[1][13851] = '63039'
y[1][12321] = '77501'
ff = np.array(y[1])
for col in range(5):
    ff2 = np.zeros(len(y[1]))
    for i in range(len(ff)):
#         print(i)
        ff2[i] = int(ff[i][col]) 
    y[str(col)] = ff2
y = y.drop([1], axis=1)
y
plt.imshow(files[0])
y.loc[0]
from sklearn.preprocessing import OneHotEncoder
for j in range(5):  
    one_data = OneHotEncoder(sparse=False)
    one_data = one_data.fit_transform(np.array(y[str(j)].tolist()).reshape(-1,1))
    for i in range(10):
        y[str(j)+'-'+str(i)] = one_data[:,i]
    y = y.drop([str(j)], axis=1)
print(one_data)
y.to_numpy().shape
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 110, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mse'])
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(files, y.to_numpy(),  
                                                  test_size=0.1, 
#                                                   stratify=np.array(y), 
                                                  random_state=42)

# Out of memory
del(files, y)
import gc
gc.collect()
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val), batch_size=512)
ii = 999
y_pred = model.predict(X_train[ii:ii+1])
print(np.argmax(y_pred[0,:10]),np.argmax(y_pred[0,10:20]),np.argmax(y_pred[0,20:30]),np.argmax(y_pred[0,30:40]),np.argmax(y_pred[0,40:]))
plt.imshow(X_train[ii])
model.load_weights('weights/207kParams')
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = "{}{}{}{}{}".format(np.argmax(predictions_array[:10]),np.argmax(predictions_array[10:20]),
                                   np.argmax(predictions_array[20:30]),np.argmax(predictions_array[30:40]),
                                   np.argmax(predictions_array[40:]))
    true_label = "{}{}{}{}{}".format(np.argmax(true_label[:10]),np.argmax(true_label[10:20]),
                                   np.argmax(true_label[20:30]),np.argmax(true_label[30:40]),
                                   np.argmax(true_label[40:]))
    
    if predicted_label == true_label:
        color = 'blue'
    else:
        print('predicted:',predicted_label, 'true:', true_label)
        color = 'red'
    
    plt.xlabel("{}{}{}{}{}".format(np.argmax(predictions_array[:10]),np.argmax(predictions_array[10:20]),
                                   np.argmax(predictions_array[20:30]),np.argmax(predictions_array[30:40]),
                                   np.argmax(predictions_array[40:])),
                                  color=color)
predictions = model.predict(X_val)
    
i = 10
rows = 6
cols = 6
plt.figure(figsize=(cols*3.5,rows*1.5))
for k in range(rows*cols):
    
    plt.subplot(rows,cols,k+1)
    plot_image(i+k, predictions[i+k], y_val, X_val)

plt.show()