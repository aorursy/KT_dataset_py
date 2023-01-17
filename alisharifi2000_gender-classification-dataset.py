# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization

from keras.preprocessing.image import ImageDataGenerator ,img_to_array, load_img

from tensorflow.keras.callbacks import EarlyStopping

from keras.optimizers import SGD

from keras.utils import plot_model

from keras import Model

from sklearn.metrics import confusion_matrix
# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir="/kaggle/input/gender-classification-dataset/Training"

val_dir="/kaggle/input/gender-classification-dataset/Validation"



train_dir_male = train_dir + '/male'

train_dir_female = train_dir + '/female'



val_dir_male  = val_dir + '/male'

val_dir_female  = val_dir + '/female'
print('number of male training images - ',len(os.listdir(train_dir_male)))

print('number of female training images - ',len(os.listdir(train_dir_female)))

print('----------------------------------------------------------------------')

print('number of normal validation  images - ',len(os.listdir(val_dir_male)))

print('number of pneumonia validation  images - ',len(os.listdir(val_dir_female)))
data_generator = ImageDataGenerator(rescale= 1./255 , validation_split=0.2)
batch_size = 64



training_data = data_generator.flow_from_directory(directory = train_dir,

                                                   target_size = (64, 64),

                                                   class_mode='binary',

                                                   color_mode= "rgb",

                                                   subset='training',

                                                   batch_size = batch_size)



testing_data = data_generator.flow_from_directory(directory = train_dir,

                                                  target_size = (64, 64),

                                                  class_mode='binary',

                                                  color_mode= "rgb",

                                                  subset='validation',

                                                  batch_size = batch_size)



validation_data = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (64, 64),

                                                  class_mode= None,

                                                  color_mode= "rgb",

                                                  batch_size = batch_size)





evaluation_generator = data_generator.flow_from_directory(directory = val_dir,

                                                  target_size = (64, 64),

                                                  class_mode= 'binary',

                                                  color_mode= "rgb",

                                                  batch_size = batch_size)
set(training_data.classes)
labels = (testing_data.class_indices)

print (labels)
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=2, patience=8)
input_model = Input(training_data.image_shape)





model1 = Conv2D(16,(7,7), activation='relu')(input_model)

model1 = Conv2D(32,(6,6), activation='relu', padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2,2))(model1)

model1 = Conv2D(32,(6,6), activation='relu' ,padding='same')(model1)

model1 = Conv2D(64,(5,5), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Conv2D(64,(5,5), activation='relu' ,padding='same')(model1)

model1 = Conv2D(128,(5,5), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Conv2D(256,(4,4), activation='relu' ,padding='same')(model1)

model1 = Conv2D(256,(4,4), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2, 2))(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='same')(model1)

model1 = Conv2D(512,(3,3), activation='relu' ,padding='valid')(model1)

model1 = BatchNormalization()(model1)

model1 = Flatten()(model1)

#########################################################                          

model2 = Conv2D(16,(4,4), activation='relu')(input_model)  

model2 = Conv2D(16,(4,4), activation='relu', padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = MaxPooling2D((3, 3))(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2) 

model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same')(model2)

model2 = Conv2D(64,(2,2), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(128,(1,1), activation='relu' ,padding='same')(model2)

model2 = Conv2D(128,(1,1), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(256,(1,1), activation='relu' ,padding='same')(model2)

model2 = Conv2D(512,(1,1), activation='relu' ,padding='valid')(model2)

model2 = BatchNormalization()(model2)

model2 = Flatten()(model2)

########################################################

merged = Concatenate()([model1, model2])

merged = Dense(units = 512, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

merged = Dropout(rate = 0.2)(merged)

merged = Dense(units = 64, activation = 'relu')(merged)

merged = Dense(units = 32, activation = 'relu')(merged)

merged = Dense(units = 16, activation = 'relu')(merged)

merged = Dense(units = 8, activation = 'relu')(merged)

merged = Dense(units = 4, activation = 'relu')(merged)

merged = Dense(units = 2, activation = 'relu')(merged)

output = Dense(activation = 'sigmoid', units = 1)(merged)



model = Model(inputs= [input_model], outputs=[output])
sgd = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
history =  model.fit_generator(generator = training_data,

                               epochs = 35,

                               #steps_per_epoch = int(len(training_data)/batch_size),

                               validation_data = testing_data ,

                               #validation_steps = int(len(testing_data)/batch_size),

                               callbacks=[es],

                               verbose=1)
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
#evaluate the model

scores = model.evaluate_generator(evaluation_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
pred = model.predict_generator(validation_data)

print(pred.shape)
pred = pred.reshape(1,pred.shape[0])

predicted_class_indices= np.round_(pred)

labels = (validation_data.class_indices)

print(predicted_class_indices)

print (labels)
true_labels = []

perdict_labels = predicted_class_indices[0]



for i in range(len(glob.glob(val_dir_female +'/*'))):

    true_labels.append(0)

for i in range(len(glob.glob(val_dir_male +'/*'))):

    true_labels.append(1)
cm = confusion_matrix(true_labels, perdict_labels)

sns.heatmap(cm, fmt='4',annot=True).set(ylabel="True Label", xlabel="Predicted Label")

plt.show()

plt.savefig('confusion_matrix.jpg')
sns.heatmap(cm/np.sum(cm), annot=True, 

            fmt='.2%').set(ylabel="True Label", xlabel="Predicted Label")

plt.show()

plt.savefig('confusion_matrix_percentage.jpg')
paths = glob.glob(val_dir_female +'/*')

for i in range(0,10):

    test_image = image.load_img(paths[i], target_size = (64, 64))

    plt.imshow(test_image)

    if predicted_class_indices[0][i] == 0:

        pred_label = 'female'

    else:

        pred_label = 'male'

    

    print('True Label female - Perdict Label : {}'.format(pred_label))

    labels = (training_data.class_indices)

    print (labels)

    plt.show()
paths = glob.glob(val_dir_male +'/*')

l = len(glob.glob(val_dir_female +'/*'))

for i in range(0,10):

    test_image = image.load_img(paths[i], target_size = (64, 64))

    plt.imshow(test_image)

    if predicted_class_indices[0][l+i] == 0:

        pred_label = 'female'

    else:

        pred_label = 'male'

    

    print('True Label male - Perdict Label : {}'.format(pred_label))

    labels = (training_data.class_indices)

    print (labels)

    plt.show()