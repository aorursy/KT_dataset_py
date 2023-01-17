import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, tqdm_notebook
from keras import models
from keras import layers
from keras import optimizers
train_dir = "../input/train/train/"
test_dir = "../input/test/test/"
file_df = pd.read_csv("../input/train.csv")

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',
                        input_shape = (32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])
model.summary()
x_train = []
y_train = []
images = file_df['id'].values
for image_id in tqdm_notebook(images):
    x_train.append(cv2.imread(train_dir + image_id))    
    y_train.append(file_df[file_df['id'] == image_id]['has_cactus'].values[0])  

#preprocessing images
x_train = np.asarray(x_train)
x_train = x_train.astype('float32')
x_train /= 255
y_train = np.asarray(y_train)

%%time
# Train model
history = model.fit(x_train, y_train,
              batch_size=32,
              epochs=100,
              validation_split=0.2,
              shuffle=True,)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
%%time
x_test = []
test_images = []
for image_id in tqdm_notebook(os.listdir(test_dir)):
    x_test.append(cv2.imread(test_dir + image_id))     
    test_images.append(image_id)
x_test = np.asarray(x_test)
x_test = x_test.astype('float32')
x_test /= 255
from keras.preprocessing import image
n_id  = np.array([],dtype = np.object)
n_has = np.array([],dtype = np.object)
for i in os.listdir(test_dir):
    n_id = np.append(n_id, i)
    test_image = image.load_img(test_dir+i, target_size = (32,32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    n_has      = np.append(n_has,model.predict(test_image))
sub = pd.DataFrame({'id':n_id,'has_cactus':n_has})
sub.to_csv("submission.csv",index = False)
    

