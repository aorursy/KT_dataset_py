import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE=8
EPOCHS=25
IMAGE_HEIGHT=64
IMAGE_WIDTH=64

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/tr',
                                                 target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/vl',
                                            target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')
score=model.fit_generator(training_set,
                         steps_per_epoch=503,
                         epochs=EPOCHS,
                         validation_data=test_set,
                         validation_steps=38
                         )
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,9))
ax1.plot(score.history['loss'],'r',linewidth=1.0)
ax1.plot(score.history['val_loss'],'b',linewidth=1.0)
ax1.legend(['Training loss', 'Validation Loss'],fontsize=12)
ax1.set_xlabel('Epochs ',fontsize=16)
ax1.set_ylabel('Loss',fontsize=16)
ax1.set_title('Loss Curves',fontsize=16)
ax1.grid(True)
# Accuracy Curves
ax2.plot(score.history['acc'],'r',linewidth=1.0)
ax2.plot(score.history['val_acc'],'b',linewidth=1.0)
ax2.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=12)
ax2.set_xlabel('Epochs ',fontsize=16)
ax2.set_ylabel('Accuracy',fontsize=16)
ax2.set_title('Accuracy Curves',fontsize=16)
ax2.grid(True)
from keras.preprocessing import image

test_image = image.load_img('dataset/sample/tulip_sample.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'Daisy'
elif result[0][1]==1:
    prediction = 'Dandelion'
elif result[0][2]==1:
    prediction = 'Rose'
elif result[0][3]==1:
    prediction='Sunflower'
elif result[0][4]==1:
    prediction='Tulip'

print('I guess this flower is %s' %prediction)