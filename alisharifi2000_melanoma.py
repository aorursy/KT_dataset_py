# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Conv2D,SeparableConv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization

from keras.preprocessing.image import ImageDataGenerator ,img_to_array, load_img

from keras.callbacks import EarlyStopping

from keras.optimizers import SGD

from keras.utils import plot_model

from keras import Model

from sklearn.metrics import confusion_matrix
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
test_dir="/kaggle/input/melanoma/DermMel/test"

train_dir="/kaggle/input//melanoma/DermMel/train_sep"

val_dir="/kaggle/input//melanoma/DermMel/valid"



train_dir_noraml = train_dir + '/Melanoma'

train_dir_melanoma = train_dir + '/NotMelanoma'



test_dir_noraml  = test_dir + '/Melanoma'

test_dir_melanoma  = test_dir + '/NotMelanoma'



val_dir_noraml  = val_dir + '/Melanoma'

val_dir_melanoma  = val_dir + '/NotMelanoma'
print('number of normal training images - ',len(os.listdir(train_dir_noraml)))

print('number of pneumonia training images - ',len(os.listdir(train_dir_melanoma)))

print('----------------------------------------------------------------------')

print('number of normal testing  images - ',len(os.listdir(test_dir_noraml)))

print('number of pneumonia testing  images - ',len(os.listdir(test_dir_melanoma)))

print('----------------------------------------------------------------------')

print('number of normal validation  images - ',len(os.listdir(val_dir_noraml)))

print('number of pneumonia validation  images - ',len(os.listdir(val_dir_melanoma)))
data_generator = ImageDataGenerator(rescale= 1./255 ,

                                    rotation_range=10,

                                    width_shift_range=0.1,

                                    height_shift_range=0.1,

                                    shear_range = 0.1,

                                    zoom_range = 0.1)
batch_size = 64

training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   target_size = (64, 64),

                                                   class_mode='binary',

                                                   color_mode= "rgb",

                                                   batch_size = batch_size)



testing_data = data_generator.flow_from_directory(directory = test_dir,

                                                  target_size = (64, 64),

                                                  class_mode='binary',

                                                  color_mode= "rgb",

                                                  batch_size = batch_size)



test_generator = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (64, 64),

                                                  class_mode=None,

                                                  color_mode= "rgb",

                                                  batch_size = batch_size)



evaluation_generator = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (64, 64),

                                                  class_mode= 'binary',

                                                  color_mode= "rgb",

                                                  batch_size = batch_size)
labels = (testing_data.class_indices)

print (labels)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=7)
input_model = Input(training_data.image_shape , name = 'ImageInput')





model1 = Conv2D(16,(7,7), activation='relu',name = 'Conv7-1-1')(input_model)

model1 = Conv2D(32,(6,6), activation='relu', padding='same',name = 'Conv6-1-2')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2,2),name = 'pool2-1-1')(model1)

model1 = Conv2D(32,(6,6), activation='relu' ,padding='same',name = 'Conv6-1-3')(model1)

model1 = Conv2D(64,(5,5), activation='relu' ,padding='same',name = 'Conv5-1-4')(model1)

model1 = BatchNormalization(name = 'Bnorm1-1')(model1)

model1 = AveragePooling2D((2, 2),name = 'pool2-1-2')(model1)

model1 = Conv2D(64,(5,5), activation='relu' ,padding='same',name = 'Conv5-1-5')(model1)

model1 = Conv2D(128,(5,5), activation='relu' ,padding='same',name = 'Conv5-1-6')(model1)

model1 = BatchNormalization(name = 'Bnorm1-2')(model1)

model1 = AveragePooling2D((2, 2),name = 'pool2-1-3')(model1)

model1 = Conv2D(256,(4,4), activation='relu' ,padding='same',name = 'Conv4-1-7')(model1)

model1 = Conv2D(256,(4,4), activation='relu' ,padding='same',name = 'Conv4-1-8')(model1)

model1 = BatchNormalization(name = 'Bnorm1-3')(model1)

model1 = MaxPooling2D((2, 2),name = 'pool2-1-4')(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='same',name = 'Conv3-1-9')(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='valid',name = 'Conv3-1-10')(model1)

model1 = BatchNormalization(name = 'Bnorm1-4')(model1)

model1 = Flatten(name = 'flat1')(model1)

#########################################################                          

model2 = Conv2D(16,(4,4), activation='relu',name = 'Conv4-2-1')(input_model)  

model2 = Conv2D(16,(4,4), activation='relu', padding='same',name = 'Conv4-2-2')(model2)

model2 = BatchNormalization(name = 'Bnorm2-1')(model2)

model2 = MaxPooling2D((3, 3),name = 'pool3-2-1')(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same',name = 'Conv3-2-3')(model2) 

model2 = Conv2D(32,(3,3), activation='relu', padding='same',name = 'Conv3-2-4')(model2)

model2 = BatchNormalization(name = 'Bnorm2-2')(model2)

model2 = AveragePooling2D((2, 2),name = 'pool2-2-2')(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same',name = 'Conv3-2-5')(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same',name = 'Conv2-2-6')(model2)

model2 = BatchNormalization(name = 'Bnorm2-3')(model2)

model2 = AveragePooling2D((2, 2),name = 'pool2-2-3')(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same',name = 'Conv2-2-7')(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same',name = 'Conv2-2-8')(model2)

model2 = BatchNormalization(name = 'Bnorm2-4')(model2)

model2 = AveragePooling2D((2, 2),name = 'pool2-2-4')(model2)

model2 = Conv2D(128,(1,1), activation='relu' ,padding='same',name = 'Conv1-2-9')(model2)

model2 = Conv2D(128,(1,1), activation='relu' ,padding='same',name = 'Conv1-2-10')(model2)

model2 = BatchNormalization(name = 'Bnorm2-5')(model2)

model2 = AveragePooling2D((2, 2),name = 'pool2-2-5')(model2)

model2 = Conv2D(256,(1,1), activation='relu' ,padding='same',name = 'Conv1-2-11')(model2)

model2 = Conv2D(512,(1,1), activation='relu' ,padding='valid',name = 'Conv1-2-12')(model2)

model2 = BatchNormalization(name = 'Bnorm2-6')(model2)

model2 = Flatten(name = 'flat2')(model2)

########################################################

model3 = SeparableConv2D(16,(9,9),activation='relu',name = 'SepConv9-3-1')(input_model) 

model3 = Conv2D(16,(9,9),activation='relu',padding = 'same',name = 'Conv9-3-2')(model3)

model3 = BatchNormalization(name = 'Bnorm3-1')(model3)

model3 = MaxPooling2D((2, 2),name = 'pool2-3-1')(model3)

model3 = SeparableConv2D(16,(7,7),activation='relu',padding = 'same',name = 'SepConv7-3-3')(model3) 

model3 = Conv2D(16,(7,7),activation='relu',padding = 'same',name = 'Conv7-3-4')(model3)

model3 = BatchNormalization(name = 'Bnorm3-2')(model3)

model3 = MaxPooling2D((2, 2),name = 'pool2-3-2')(model3)

model3 = SeparableConv2D(32,(5,5),activation='relu',padding = 'valid',name = 'SepConv5-3-5')(model3) 

model3 = Conv2D(64,(5,5),activation='relu',padding = 'valid',name = 'SepConv5-3-6')(model3)

model3 = Conv2D(128,(3,3),activation='relu',padding = 'valid',name = 'Conv3-3-7')(model3) 

model3 = SeparableConv2D(128,(3,3),activation='relu',padding = 'valid',name = 'SepConv3-3-7')(model3)

model3 = BatchNormalization(name = 'Bnorm3-3')(model3)

model3 = MaxPooling2D((2, 2),name = 'pool2-3-3')(model3)

model3 = Flatten(name = 'flat3')(model3)

########################################################

merged = Concatenate()([model1, model2, model3])

merged = Dense(units = 512, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

merged = Dropout(rate = 0.2)(merged)

merged = Dense(units = 64, activation = 'relu')(merged)

merged = Dense(units = 8, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

merged = Dropout(rate = 0.15)(merged)

merged = Dense(units = 2, activation = 'relu')(merged)

output = Dense(activation = 'sigmoid', units = 1)(merged)



model = Model(inputs= [input_model], outputs=[output])
sgd = SGD(lr=0.01, momentum=0.9 ,nesterov = True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
history_pre = model.fit_generator(training_data,epochs = 2,validation_data = testing_data ,callbacks=[es],verbose=1)
history =  model.fit_generator(training_data,epochs = 20,validation_data = testing_data ,callbacks=[es],verbose=1)
model.save_weights("weights.h5")
val_loss = history.history['val_loss']

loss = history.history['loss']



plt.plot(val_loss)

plt.plot(loss)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Val error','Train error'], loc='upper right')

plt.savefig('plot_pre_error.png')

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
#evaluate the model

scores = model.evaluate_generator(evaluation_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
pred = model.predict_generator(test_generator)

print(pred.shape)
pred = pred.reshape(1,pred.shape[0])

predicted_class_indices= np.round_(pred)

labels = (test_generator.class_indices)

print(predicted_class_indices)

print (labels)
true_labels = []

perdict_labels = predicted_class_indices[0]



for i in range(len(glob.glob(val_dir_melanoma +'/*'))):

    true_labels.append(0)

    

for i in range(len(glob.glob(val_dir_noraml +'/*'))):

    true_labels.append(1)
cm = confusion_matrix(true_labels, perdict_labels)

sns.heatmap(cm, fmt='4',annot=True).set(ylabel="True Label", xlabel="Predicted Label")

plt.show()

plt.savefig('confusion_matrix.jpg')
sns.heatmap(cm/np.sum(cm), annot=True, 

            fmt='.2%').set(ylabel="True Label", xlabel="Predicted Label")

plt.show()

plt.savefig('confusion_matrix_percentage.jpg')
paths = glob.glob(val_dir_noraml +'/*')

for i in range(0,10):

    test_image = image.load_img(paths[i], target_size = (64, 64))

    plt.imshow(test_image)

    if predicted_class_indices[0][i] == 0:

        pred_label = 'Melanoma'

    else:

        pred_label = 'Normal'

    

    print('True Label Melanoma - Perdict Label : {}'.format(pred_label))

    labels = (training_data.class_indices)

    print (labels)

    plt.show()