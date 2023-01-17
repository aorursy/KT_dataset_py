import keras
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1/255)

test_datagen = ImageDataGenerator(rescale = 1/255)
trainset = train_datagen.flow_from_directory(r"../input/catsdogs/training_set/training_set",target_size = (150,150),class_mode='binary',batch_size=80)

testset = test_datagen.flow_from_directory(r"../input/catsdogs/test_set/test_set",target_size = (150,150),class_mode = 'binary',batch_size= 20)
model = keras.models.Sequential([Conv2D(16,(3,3),input_shape = (150,150,3),activation = 'relu'),

                                 MaxPooling2D(2,2),

                                 Conv2D(32,(3,3),activation = 'relu'),

                                 MaxPooling2D(2,2),

                                 Conv2D(64,(3,3),activation = 'relu'),

                                 MaxPooling2D(2,2),

                                 Flatten(),

                                 Dense(512,activation = 'relu'),

                                 Dense(1,activation = 'sigmoid')

    

])
model.summary()
from keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr = 0.001),loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit_generator(trainset,steps_per_epoch=50,epochs = 10,validation_data=testset,validation_steps=20)
trainset.class_indices
from keras.preprocessing import image

testdata1 = image.load_img("../input/cats-vs-dogs-testing/dogeyes.jpg",target_size = (150,150))

import numpy as np

x1=image.img_to_array(testdata1)

x1=np.expand_dims(x1, axis=0)

images = np.vstack([x1])

ans = model.predict(images)

if ans[0][0] == 0:

    print("it is a CAT")

else:

    print("it is a DOG")