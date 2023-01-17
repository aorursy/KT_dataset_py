# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import Model
test_dir="../input/dogs-cats-images/dog vs cat/dataset/test_set"
train_dir="../input/dogs-cats-images/dog vs cat/dataset/training_set"

train_dir_cats = train_dir + '/cats'
train_dir_dogs = train_dir + '/dogs'
test_dir_cats = test_dir + '/cats'
test_dir_dogs = test_dir + '/dogs'
print('number of cats training images - ',len(os.listdir(train_dir_cats)))
print('number of dogs training images - ',len(os.listdir(train_dir_dogs)))
print('number of cats testing images - ',len(os.listdir(test_dir_cats)))
print('number of dogs testing images - ',len(os.listdir(test_dir_dogs)))
data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.1)
batch_size = 64
training_data = data_generator.flow_from_directory(directory = train_dir,
                                                   target_size = (32, 32),
                                                   batch_size = batch_size,
                                                   class_mode = 'binary')
testing_data = data_generator.flow_from_directory(directory = test_dir,
                                                  target_size = (32, 32),
                                                  batch_size = batch_size,
                                                  class_mode = 'binary')
input_model = Input(training_data.image_shape)

model1 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(input_model)
model1 = MaxPooling2D(pool_size = (2, 2))(model1)
model1 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(model1)
model1 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu')(model1)
model1 = MaxPooling2D(pool_size = (2, 2))(model1)
model1 = Conv2D(filters = 32, kernel_size = (2, 2), activation = 'relu')(model1)
model1 = AveragePooling2D(pool_size = (2, 2))(model1)
model1 = Flatten()(model1)
########################
model2 = Conv2D(filters = 128, kernel_size = (4, 4), activation = 'relu')(input_model)
model2 = MaxPooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu')(model2)
model2 = MaxPooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 64, kernel_size = (2, 2), activation = 'relu')(model2)
model2 = AveragePooling2D(pool_size = (2, 2))(model2)
model2 = Flatten()(model2)
########################
merged = Concatenate()([model1, model2])
merged = Dense(units = 256, activation = 'relu')(merged)
merged = Dropout(rate = 0.2)(merged)
merged = Dense(units = 16, activation = 'relu')(merged)
merged = Dense(units = 8, activation = 'relu')(merged)
output = Dense(units = len(set(training_data.classes)), activation = 'softmax')(merged)

model = Model(inputs= [input_model], outputs=[output])
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.summary()
plot_model(model, show_shapes=True)
history =  model.fit_generator(training_data,epochs = 100,validation_data = testing_data,callbacks=[es],verbose=1)
model.save_weights("weights.h5")
val_loss = history.history['val_loss']
loss = history.history['loss']

plt.plot(val_loss)
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Val error','Train error'], loc='upper right')
plt.savefig('plot_error.png')
plt.show()
val_accuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']

plt.plot(val_accuracy)
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['Val accuracy','Train accuracy'], loc='upper right')
plt.savefig( 'plot_accuracy.png')
plt.show()
# testing the model
def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (32, 32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    return prediction
print(testing_image(test_dir + '/cats/cat.4003.jpg'))
plt.imshow(image.load_img((test_dir + '/cats/cat.4003.jpg')))
plt.imshow(image.load_img((test_dir + '/cats/cat.4003.jpg'),target_size = (64, 64)))
