# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential, load_model

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense, Dropout, Activation

from PIL import Image

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from keras import optimizers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
input_shape = (224, 224, 3)



# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#classifier.add(Dropout(0.2))



# Adding a second convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#classifier.add(Dropout(0.2))



# Adding a third convolutional layer

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#classifier.add(Dropout(0.2))



# Adding a fourth convolutional layer

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#classifier.add(Dropout(0.2))







classifier.add(Conv2D(128, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))







# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 64, activation = 'relu'))

#classifier.add(Dropout(0.5))

classifier.add(Dense(units = 1, activation = 'sigmoid')) #USE SOFTMAX -> we set softmax because it models the fact that labels are mutually exclusive (one and only one of them is right).



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #USE categorical crossentropy

#vgg16_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#plot_model(classifier, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

#display(Image.open('cnn_model.png'))
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rotation_range=0,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True,

                                   vertical_flip = True)



valid_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale = 1./255)



bonus_test_datagen = ImageDataGenerator(rescale = 1./255)







training_set = train_datagen.flow_from_directory('../input/figuredetectiondataset10000/figure-dataset-10000/train/',

                                                 target_size = (128, 128),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



valid_set = valid_datagen.flow_from_directory('../input/figuredetectiondataset10000/figure-dataset-10000/valid',

                                            target_size = (128, 128),

                                            batch_size = 32,

                                            class_mode = 'binary')





test_set = test_datagen.flow_from_directory('../input/figuredetectiondataset10000/figure-dataset-10000/test',

                                            target_size = (128, 128),

                                            batch_size = 1,

                                            class_mode = 'binary',

                                            shuffle = False)



bonus_test_set = bonus_test_datagen.flow_from_directory('../input/bonustest/bonus-test',

                                            target_size = (128, 128),

                                            batch_size = 1,

                                            class_mode = 'binary',

                                            shuffle = False)
filepath = "best_model.hdf5"#"model_{acc}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#For the very first training process, after we create the first model, we load the latest model and continue training if necessary

'''history = classifier.fit_generator(training_set,

                         steps_per_epoch = 200, #200

                         epochs = 10,

                         validation_data = valid_set,

                         validation_steps = 45, #45

                         callbacks = [checkpoint])

'''
#print(history.history.keys())

print(os.listdir("/"))
#import os

#print(os.listdir('../input/model-1h5'))
#classifier.save('my_model_8.h5')

#saved_model = classifier



#saved_model = load_model('../input/models/model_2_Epochs_50_60.h5')

'''

history = saved_model.fit_generator(training_set,

                         steps_per_epoch = 200, #200

                         epochs = 10,

                         validation_data = valid_set,

                         validation_steps = 45, #45

                         callbacks = [checkpoint])



'''


#saved_model.save('model_4_Epochs_60_85.h5')





#saved_model = classifier



#history = load_model('/kaggle/working/model_12.h5')
#print(history)
'''

# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()'''
#classifier.evaluate(x_test,y_test)
import numpy as np

from keras.preprocessing import image

path = '../input/figuredetectiondataset10000/figure-dataset-10000/test/'

import random



import glob

#print(glob.glob("../input/simpledetectiondataset/Simple-Dataset/test/positive/*.jpg"))

final_image_list = []

final_image_list.append(glob.glob("../input/figuredetectiondataset10000/figure-dataset-10000/test/negative/*.jpg"))

final_image_list.append(glob.glob("../input/figuredetectiondataset10000/figure-dataset-10000/test/positive/*.jpg"))



#final_image_list = final_image_list[0] 



#print(final_image_list)



'''

random_class_id = np.random.randint(2)



class_id = 1



random_image_id = np.random.randint(750)



#image_list[random_image_id]

print(random_class_id)

print(random_image_id)

print(len(final_image_list))

random_path = final_image_list[random_class_id][random_image_id]     



    

print(random_path)



test_image = image.load_img(random_path, target_size = (128, 128))



test_image = image.img_to_array(test_image)



test_image = np.expand_dims(test_image, axis = 0)

print(test_image.shape)

test_image.reshape(128,128,3)

print(test_image.shape)



result = saved_model.predict(test_image)



threshold = 1





print("RESULT:")

print(result)

print(training_set.class_indices)

if result[0][0] >= threshold:

    prediction = 'positive'

else:

    prediction = 'negative'



print()



print("Model's Prediction:")

print(prediction)



test_image = image.load_img(random_path)



fig=plt.figure(figsize=(8, 8))



img = test_image

plt.imshow(img)

plt.show()

'''

'''

test_set_y = []

for i in range(1500):

    if i <=750:

        test_set_y.append(1)

    else:

        test_set_y.append(0)





STEP_SIZE_VALID=valid_set.n//valid_set.batch_size

STEP_SIZE_TEST=test_set.n //test_set.batch_size

STEP_SIZE_TRAIN=training_set.n//training_set.batch_size



STEP_SIZE_BONUS_TEST=bonus_test_set.n //bonus_test_set.batch_size



evaluation = saved_model.evaluate_generator(generator=test_set,

steps=STEP_SIZE_TEST)

#https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

print(evaluation)





#test_set.reset()

pred=saved_model.predict_generator(test_set,

steps=STEP_SIZE_TEST,

verbose=1)



predicted_class_indices=np.argmax(pred,axis=1)

'''

'''



labels = (training_set.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



filenames=test_set.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions})

results.to_csv("results.csv",index=False)



#scores = saved_model.evaluate(test_set, test_set.labels , verbose=0)

#print(scores)



test_accuracy_list = []



for label in range(2):

    for image_id in range(750):

        image_path = final_image_list[label][image_id]

        print(image_path)

        test_image = image.load_img(image_path, target_size = (128, 128))

        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis = 0)

        result = saved_model.predict(test_image)

        #print(result)

        print("Comparison:")

        print(label)

        print(result[0][0])

        

        if label == 0 and result[0][0] >= threshold:

            test_accuracy_list.append(1)

        elif label == 1 and result[0][0] >= threshold:

            test_accuracy_list.append(0)

        elif label == 0 and result[0][0] < threshold :

            test_accuracy_list.append(0)

        elif label == 1 and result[0][0] < threshold :

            test_accuracy_list.append(1)

        else:

            print("error")

            print(label)

            print(result[0][0])

    

test_accuracy = sum(test_accuracy_list) / len(test_accuracy_list)

print("Accuracy List Sum:")

print(sum(test_accuracy_list))

print("TEST ACCURACY")

print(test_accuracy)

print("Number of Test Images:")

print(len(test_accuracy_list))'''
#total_valid_result = classifier.predict_classes(valid_set)
#total_train_result = classifier.predict(training_set)

'''

print("PRED:")

final_test_list = []

print(pred[720:780])



for i in range(200):

    if i < 100 and pred[i] <= 0.5:

        final_test_list.append(1)

    elif i >= 100 and pred[i] >= 0.5:

        final_test_list.append(1)

    else:

        final_test_list.append(0)

        misprediction_path = final_image_list[i//100][i - 100* (i//100)]     

        

        test_image = image.load_img(misprediction_path)



        fig=plt.figure(figsize=(4, 4))



        img = test_image

        plt.imshow(img)

        plt.show()

        

        print(i//100)

        print(i - 100* (i//100))

        print("Wrong Prediction:")

        print(pred[i])

        

final_test_accuracy = sum(final_test_list) / len(final_test_list)

print("Accuracy List Sum:")

print(sum(final_test_list))

print("TEST ACCURACY")

print(final_test_accuracy)



'''



#print(predicted_class_indices[755:770])

#print(sum(predicted_class_indices))



#print(len(predictions))

print("THE END")