train_dir_path = '/kaggle/input/dogcat-classificationcnn/dataset/training_set/'
test_dir_path = '/kaggle/input/dogcat-classificationcnn/dataset/test_set/'
img_width =  img_height = 299 # width and height of input image (must be 299x299)
from keras.preprocessing.image import ImageDataGenerator

#Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

#Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = train_datagen.flow_from_directory(
    train_dir_path,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical',)

test_set = test_datagen.flow_from_directory(
    test_dir_path,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='categorical',)
class_label = train_set.class_indices
print(class_label)
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image 

dogs_image = os.listdir(train_dir_path+'dogs/')[:12]
cats_image = os.listdir(train_dir_path+'cats/')[:12]
images = dogs_image + cats_image
labels = list(map(lambda x: x.split('.')[0], images))
plt.figure(figsize=(13,13))
for i,name in enumerate(images):
    plt.subplot(6,4,i+1)
    plt.xticks([])
    plt.yticks([])
    img = image.load_img(train_dir_path+labels[i]+'s/'+name,target_size=(299,299))
    plt.xlabel(labels[i])
    plt.imshow(img)
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input
from keras.applications import xception

# create the base pre-trained model
base_model = xception.Xception(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=p-redictions)
# freeze all convolutional Xception layers
for layer in base_model.layers:
    layer.trainable = False

model.summary()
from keras.callbacks import ModelCheckpoint

optimizer = RMSprop(lr=0.001, rho=0.9)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=["accuracy"])

hist = model.fit_generator(
    train_set,
    steps_per_epoch= train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# we chose to train the top 2 xception blocks, i.e. we will freeze the first 116 layers and unfreeze the rest:
for layer in model.layers[:96]:
    layer.trainable = False
for layer in model.layers[96:]:
    layer.trainable = True
    
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=["accuracy"])

hist = model.fit_generator(
    train_set,
    steps_per_epoch= train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# we chose to train the top 2 xception blocks, i.e. we will freeze the first 116 layers and unfreeze the rest:
for layer in model.layers[:16]:
    layer.trainable = False
for layer in model.layers[16:]:
    layer.trainable = True
    
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=["accuracy"])

hist = model.fit_generator(
    train_set,
    steps_per_epoch= train_set.samples // train_set.batch_size,
    epochs=10,
    validation_data=test_set,
    validation_steps=test_set.samples // test_set.batch_size)
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from PIL import Image
import requests
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
import numpy as np
def load_img_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((299,299))
    return img

def print_predict(img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    class_index = np.argmax(pred)
    probability = pred[0][class_index]*100
    print(probability)
    if probability > 99:
        if class_index == 0:
            print('It is a cat')
        elif class_index == 1:
            print('It is a dog')
    else:
        print('I only detect cat and dog or try another image that have only 1 cat or dog')
img = load_img_from_url('https://vnreview.vn/image/19/87/64/1987643.jpg')
img
print_predict(img)
img = load_img_from_url('https://i.pinimg.com/originals/70/c6/46/70c6461c88417f44ddb9926577eb3fb4.jpg')
img
print_predict(img)
img = load_img_from_url('https://petviet.vn/wp-content/uploads/2018/03/1803_5-ly-do-khien-ban-muon-nuoi-meo-ngay-lap-tuc-01.jpg')
img
print_predict(img)
img = load_img_from_url('https://www.nationalgeographic.com/content/dam/animals/pictures/mammals/a/african-lion/african-lion.adapt.1900.1.JPG')
img
print_predict(img)
img = load_img_from_url('https://ss-images.catscdn.vn/wpm450/2020/05/22/7523489/son-tung-mtp-lo-anh-dung-do-doi-voi-thieu-bao-tram-9-15848102443021860636400.jpg')
img
print_predict(img)
model.save("cats_and_dogs.h5")