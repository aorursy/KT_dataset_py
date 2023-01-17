import os
print(os.listdir("../input/dataset/dataset"))
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)
from keras.layers import Conv2D,Dropout,Dense,Flatten,MaxPooling2D,ZeroPadding2D
from keras.models import Sequential
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.optimizers import SGD
model=Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss=binary_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()
training_set=train_datagen.flow_from_directory(directory='../input/dataset/dataset/training_set',target_size=(64,64),class_mode='binary')
test_set=test_datagen.flow_from_directory('../input/dataset/dataset/test_set/',target_size=(64,64),class_mode='binary')
history=model.fit_generator(training_set,steps_per_epoch = 2000//32,epochs =20 ,validation_data = test_set,validation_steps = 800//32)
