import tensorflow as tf
import multiprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, MaxPool2D
from keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
"""
!mv ../input/cat-and-dog/test_set/test_set/cats/*.jpg 
!mv ../input/cat-and-dog/test_set/test_set/dogs/*.jpg ../input/catndog/cats_and_dogs_filtered/train/dogs
!mv ../input/cat-and-dog/training_set/training_set/cats/*.jpg ../input/catndog/cats_and_dogs_filtered/validation/cats
!mv ../input/cat-and-dog/training_set/training_set/dogs/*.jpg ../input/catndog/cats_and_dogs_filtered/validation/dogs
"""
PATH = os.path.join('..', 'input', 'cat-and-dog') #'catndog','cats_and_dogs_filtered') 
train_dir = os.path.join(PATH, 'training_set', 'training_set') #'train')
validation_dir = os.path.join(PATH, 'test_set','test_set')#'validation')
# directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
# directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print('—')
print('Total training images:', total_train)
print('Total validation images:', total_val)
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1./255)
# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size, 
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(
batch_size=batch_size,
directory=validation_dir,
target_size=(IMG_HEIGHT, IMG_WIDTH),
class_mode='categorical')
sample_training_images, _ = next(train_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])

def print_even(v1,v2) :  
    for i in v1:
        for j in v2:
            yield i+j
  
# initializing list  
test_list = [1, 4, 5, 6, 7] 

# printing even numbers  
print ("The even numbers in list are : ", end = " ") 
for j in print_even(test_list, test_list): 
    print (j, end = " ")
"""
    model = Sequential([
        Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
# Create model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])



"""
acclist = []

def attempt(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4):
    model = 0

    model = Sequential([
        Conv2D(l1, l11, padding='same', activation=s1, input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(),
        Conv2D(l2, l12, padding='same', activation=s2),
        MaxPooling2D(),
        Conv2D(l3, l13, padding='same', activation=s3),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation=s4),
        Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    # Train the model
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )
    acclist.append(history.history['accuracy'][0])

#@@用很多for循环把attempt函数用各种变量的排列组合都跑一遍，找到'accuracy'最高的存入winner
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
model = 0

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2DD(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Train the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
acclist.append(history.history['accuracy'][0])
attempt(16,32,64,3,3,3,'relu','relu','relu','relu')
maxacc=0
llist = [16,32,64]
l1list = [3,5]
slist = ['relu']
for l1 in llist:
    for l2 in llist:
        for l3 in llist:
            for l11 in l1list:
                for l12 in l1list:
                    for l13 in l1list:
                        for s1 in slist: 
                            for s2 in slist: 
                                for s3 in slist: 
                                    for s4 in slist: 
                                        print(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4)
                                        attempt(l1,l2,l3,l11,l12,l13,s1,s2,s3,s4)
                                        if maxacc<(acclist[-1]):
                                            maxacc=acclist[-1]
                                            winner=[l1,l2,l3,l11,l12,l13,s1,s2,s3,s4,maxacc]
                                            print("%%")

print(winner)
# Predict
results = model.predict(sample_training_images[:5])
print(acclist)
model.save('my_model.h5')
# 重新创建完全相同的模型，包括其权重和优化程序
new_model = tf.keras.models.load_model('my_model.h5')
# 显示网络结构
new_model.summary()
!pip install tensorflowjs
!tensorflowjs_converter --input_format=keras ./my_model.h5 tfjs_model
!zip -r downloadmeeeee.zip tfjs_model
!ls -sh