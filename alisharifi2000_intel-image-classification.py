# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator ,img_to_array, load_img
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import Model
import glob
import random
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
test_dir="/kaggle/input/intel-image-classification/seg_test/seg_test"
train_dir="/kaggle/input/intel-image-classification/seg_train/seg_train"
val_dir="/kaggle/input/intel-image-classification/seg_pred/"

train_dir_glacier = train_dir + '/glacier'
train_dir_buildings = train_dir + '/buildings'
train_dir_sea = train_dir + '/sea'
train_dir_mountain = train_dir + '/mountain'
train_dir_street = train_dir + '/street'
train_dir_forest = train_dir + '/forest'

test_dir_glacier  = test_dir + '/glacier'
test_dir_buildings = test_dir + '/buildings'
test_dir_sea = test_dir + '/sea'
test_dir_mountain = test_dir + '/mountain'
test_dir_street = test_dir + '/street'
test_dir_forest = test_dir + '/forest'
print('number of glacier training images - ',len(os.listdir(train_dir_glacier)))
print('number of buildings training images - ',len(os.listdir(train_dir_buildings)))
print('number of sea training images - ',len(os.listdir(train_dir_sea)))
print('number of mountain training images - ',len(os.listdir(train_dir_mountain)))
print('number of street training images - ',len(os.listdir(train_dir_street)))
print('number of forest training images - ',len(os.listdir(train_dir_forest)))
print('----------------------------------------------------------------------')
print('number of glacier testing  images - ',len(os.listdir(test_dir_glacier)))
print('number of buildings testing  images - ',len(os.listdir(test_dir_buildings)))
print('number of sea training testing  - ',len(os.listdir(test_dir_sea)))
print('number of mountain testing  images - ',len(os.listdir(test_dir_mountain)))
print('number of street testing  images - ',len(os.listdir(test_dir_street)))
print('number of forest testing  images - ',len(os.listdir(test_dir_forest)))
print('----------------------------------------------------------------------')
print('number of testing  images - ',len(os.listdir(val_dir)))
img = load_img(train_dir + "/buildings/10032.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
x = img_to_array(img)
print(x.shape)
data_generator = ImageDataGenerator(rescale= 1./255)
batch_size = 64
training_data = data_generator.flow_from_directory(directory = train_dir,
                                                   target_size = (150, 150),
                                                   class_mode='categorical',
                                                   color_mode= "rgb",
                                                   batch_size = batch_size)

testing_data = data_generator.flow_from_directory(directory = test_dir,
                                                  target_size = (150, 150),
                                                  class_mode='categorical',
                                                  color_mode= "rgb",
                                                  batch_size = batch_size)

test_generator = data_generator.flow_from_directory(directory = val_dir,
                                                  target_size = (150, 150),
                                                  class_mode= None,
                                                  color_mode= "rgb",
                                                  batch_size = batch_size)
set(training_data.classes)
labels = (training_data.class_indices)
print (labels)
set(testing_data.classes)
labels = (testing_data.class_indices)
print (labels)
input_model = Input(training_data.image_shape)

model1 = Conv2D(filters = 32, kernel_size = (4, 4), activation = 'relu' , padding = 'same')(input_model)
model1 = MaxPooling2D(pool_size = (3, 3))(model1)
model1 = Conv2D(filters = 32, kernel_size = (4, 4), activation = 'relu', padding = 'same')(model1)
model1 = Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu', padding = 'same')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D(pool_size = (3, 3))(model1)
model1 = Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu', padding = 'same')(model1)
model1 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model1)
model1 = MaxPooling2D(pool_size = (2, 2))(model1)
model1 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model1)
model1 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model1)
model1 = AveragePooling2D(pool_size = (2, 2))(model1)
model1 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model1)
model1 = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model1)
model1 = BatchNormalization()(model1)
model1 = AveragePooling2D(pool_size = (2, 2))(model1)
model1 = Conv2D(filters = 256, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model1)
model1 = Conv2D(filters = 512, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model1)
model1 = BatchNormalization()(model1)
model1 = MaxPooling2D(pool_size = (2, 2))(model1)
model1 = Flatten()(model1)
########################
model2 = Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same')(input_model)
model2 = Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same')(model2)
model2 = BatchNormalization()(model2)
model2 = MaxPooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 64, kernel_size = (4, 4), activation = 'relu', padding = 'same')(model2)
model2 = MaxPooling2D(pool_size = (3, 3))(model2)
model2 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model2)
model2 = MaxPooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 128, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 128, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 128, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model2)
model2 = BatchNormalization()(model2)
model2 = AveragePooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 128, kernel_size = (2, 2), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 256, kernel_size = (1, 1), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 256, kernel_size = (1, 1), activation = 'relu', padding = 'same')(model2)
model2 = AveragePooling2D(pool_size = (2, 2))(model2)
model2 = Conv2D(filters = 512, kernel_size = (1, 1), activation = 'relu', padding = 'same')(model2)
model2 = Conv2D(filters = 512, kernel_size = (1, 1), activation = 'relu', padding = 'same')(model2)
model2 = BatchNormalization()(model2)
model2 = AveragePooling2D(pool_size = (2, 2))(model2)
model2 = Flatten()(model2)
########################
merged = Concatenate()([model1, model2])
merged = Dense(units = 512, activation = 'relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(rate = 0.2)(merged)
merged = Dense(units = 64, activation = 'relu')(merged)
merged = Dense(units = 16, activation = 'relu')(merged)
output = Dense(units = len(set(training_data.classes)), activation = 'softmax')(merged)

model = Model(inputs= [input_model], outputs=[output])
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
model.summary()
plot_model(model, show_shapes=True)
history =  model.fit_generator(training_data,epochs = 60,validation_data = testing_data ,callbacks=[es],verbose=1)
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
plt.legend(['Val accuracy','Train accuracy'], loc='lower right')
plt.savefig( 'plot_accuracy.png')
plt.show()
pred = model.predict_generator(test_generator)
print(pred.shape)
print('---------------------------------------')
print(pred)
predicted_class_indices= np.argmax(pred,axis = 1)
print('---------------------------------------')
print(predicted_class_indices)
paths = glob.glob(val_dir + 'seg_pred/*.jpg')
for i in [0, 50 , 500 , 1200, 3500 , 5000, 6400, 7300 , 7301]:
    test_image = image.load_img(paths[i], target_size = (150, 150))
    plt.imshow(test_image)
    
    if predicted_class_indices[i] == 0:
        pred_label = 'buildings'
    elif predicted_class_indices[i] == 1:
        pred_label = 'forest'
    elif predicted_class_indices[i] == 2:
        pred_label = 'glacier'
    elif predicted_class_indices[i] == 3:
        pred_label = 'mountain'
    elif predicted_class_indices[i] == 4:
        pred_label = 'sea'
    elif predicted_class_indices[i] == 5:
        pred_label = 'street'

    
    print('Perdict Label : {}'.format(pred_label))
    labels = (training_data.class_indices)
    print (labels)
    plt.show()
