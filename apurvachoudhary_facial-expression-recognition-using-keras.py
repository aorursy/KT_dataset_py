# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os
classes = 5
row, col = 48, 48
batch_size = 32
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range = 30,
                shear_range = 0.3,
                zoom_range = 0.3,
                width_shift_range = 0.4,
                height_shift_range = 0.4,
                horizontal_flip = True,
                fill_mode = 'nearest'
                )

validation_datagen = ImageDataGenerator(
                    rescale = 1./255
                    )

input_path = '../input/facial-expression-database/'

train_dataset = train_datagen.flow_from_directory(
                input_path + 'train/train/',
                color_mode = 'grayscale',
                target_size = (row, col),
                batch_size = batch_size,
                class_mode = 'categorical',
                shuffle = True
                )

validation_dataset = train_datagen.flow_from_directory(
                    input_path + 'validation/validation/',
                    color_mode = 'grayscale',
                    target_size = (row, col),
                    batch_size = batch_size,
                    class_mode = 'categorical',
                    shuffle = True
                    )
#Building the model

model = Sequential()

# Block_1
model.add(Conv2D(32, (3,3), kernel_initializer = 'he_normal',padding = 'same', input_shape = (row, col,1)))
model.add(Activation ('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), kernel_initializer = 'he_normal',padding = 'same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#print(model.get_weights())

#Block_2
model.add(Conv2D(64, (3,3),kernel_initializer = 'he_normal', padding = 'same'))
model.add(Activation ('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3),kernel_initializer = 'he_normal', padding = 'same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#print(model.get_weights())

#Block_3
model.add(Conv2D(128, (3,3),kernel_initializer = 'he_normal', padding = 'same'))
model.add(Activation ('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3),kernel_initializer = 'he_normal', padding = 'same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#print(model.get_weights())

#Block_4
model.add(Conv2D(256, (3,3),kernel_initializer = 'he_normal',padding = 'same'))
model.add(Activation ('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3),kernel_initializer = 'he_normal',padding = 'same'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#Block_5
model.add(Flatten())
model.add(Dense(128, kernel_initializer = 'he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block_6
model.add(Dense(64, kernel_initializer = 'he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Block_7
model.add(Dense(classes))
model.add(Activation('softmax'))

#print(model.get_weights())
print(model.summary())
from keras.optimizers import Adam, RMSprop, SGD

opt = Adam(learning_rate=0.001)
model.compile(optimizer = opt,
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Facial_Expression_Recognition.hdf5',
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1)


earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 9,
                          verbose = 1,
                          restore_best_weights = True)


reduceLR = ReduceLROnPlateau(monitor = 'val_loss',
                             factor = 0.2,
                             patience = 3,
                             verbose = 1,
                             min_delta = 0.0001)

model_callbacks = [checkpoint, earlystop, reduceLR]
                         
    

expression = model.fit_generator(train_dataset,
                    steps_per_epoch = train_dataset.n//batch_size,
                    epochs = 50,
                    callbacks = model_callbacks,
                    validation_data = validation_dataset,
                    validation_steps = validation_dataset.n//batch_size)                           
                    
plt.figure(1, figsize = (15,8)) 
    
plt.subplot(211)  
plt.plot(expression.history['accuracy'])  
plt.plot(expression.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 


plt.subplot(212)  
plt.plot(expression.history['loss'])  
plt.plot(expression.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

classifier = load_model('/kaggle/working/Facial_Expression_Recognition.hdf5')
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

from skimage import io
img = image.load_img('../input/testimage2/4.jpg', color_mode = "grayscale", target_size=(48, 48))
show_img=image.load_img('../input/testimage2/4.jpg', grayscale=False, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

pred = classifier.predict(x)[0]
label = class_labels[pred.argmax()]

plt.imshow(show_img)
print('Expression Prediction:',label)
face_classifier = cv2.CascadeClassifier('../input/haar-cascades-for-face-detection/haarcascade_frontalface_default.xml')
classifier = load_model('/kaggle/working/Facial_Expression_Recognition.hdf5')

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

img = cv2.imread('../input/img-test/2.jpg')
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(gray, (x,y),(x+w, y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48,48))
    
    roi = roi_gray.astype('float')/255
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis =0)
            
    pred = classifier.predict(roi)[0]
    label = class_labels[pred.argmax()]
    label_position = (x,y)
    cv2.put_Text(gray, label, label_position, (255,0,0), 3)
    
    cv2.imshow("Emotion", gray)
        
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier = load_model('/kaggle/working/Facial_Expression_Recognition.hdf5')

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1, 1.3)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(gray, (x,y),(x+w, y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        
        if np.sum(roi_gray) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis =0)
            
            pred = classifier.predict(roi)[0]
            label = class_labels[pred.argmax()]
            label_position = (x,y)
            cv2.put_Text(gray, label, label_position, (255,0,0), 3)
        else:
            cv2.put_Text(frame, "no face", (20,60), (255,0,0), 3)
            
        cv2.imshow("Emotion", gray)
        
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
            