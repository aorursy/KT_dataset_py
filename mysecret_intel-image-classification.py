import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

import os

from keras.applications import VGG16
test_dir="../input/intel-image-classification/seg_test/seg_test"

train_dir="../input/intel-image-classification/seg_train/seg_train"
img_width, img_height = 128, 128 
# Initializing the CNN

classifier = Sequential()
# Step 1 - Convolution

classifier.add(Conv2D(32,(3,3), input_shape = (128,128,3), activation = "relu"))
# Step 2 - Maxpooling

classifier.add(MaxPooling2D(pool_size = (3,3)))
# Step 3 - Flattening

classifier.add(Flatten())
# Step 4 - Full Connection to the ANN

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 6, activation = 'sigmoid'))
# Step 5 - Compiling

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Step 6 - Fitting CNN to the training set

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,target_size=(128, 128),batch_size=32,class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_dir,target_size=(128, 128),batch_size=32,class_mode='categorical')
classify_model = classifier.fit_generator(training_set, steps_per_epoch=2000, epochs=5, validation_data = test_set, validation_steps=60)
# Plot results

acc = classify_model.history['accuracy']

val_acc = classify_model.history['val_accuracy']

loss = classify_model.history['loss']

val_loss = classify_model.history['val_loss']



epochs = range(1, len(acc)+1)



plt.plot(epochs, acc, 'g', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
# from keras.preprocessing import image

# def prediction(img_path):

#     org_img = image.load_img(img_path)

#     img = image.load_img(img_path, target_size=(img_width, img_height))

#     img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range

#     img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

#     plt.imshow(org_img)                           

#     plt.axis('off')

#     plt.show()





# #     # Extract features

#     features = classifier.predict(img_tensor.reshape(1,img_width, img_height, 3))



#     # Make prediction

#     try:

#         prediction = classifier.predict(features)

#     except:

# #         prediction = classifier.predict(features.reshape(1, 1*2*3))

#         prediction = classifier.predict(features.reshape(1,64*64*3))

        

#     classes = ["buildings", "forest", "glacier", "mountains", "sea", "street"]

#     print("I see..."+str(classes[np.argmax(np.array(prediction[0]))]))
# pred_dir = "../input/intel-image-classification/seg_pred/seg_pred/"

# import random

# pred_files = random.sample(os.listdir(pred_dir),6)

# for f in pred_files:

#     prediction(pred_dir+f)
from sklearn.utils import shuffle

def data_making_for_prediction(directory):

    data=[]

    for img in os.listdir(directory):    

        path_img=os.path.join(directory,img)

        img_data=cv2.resize(cv2.imread(path_img),(128,128))

        data.append((img_data))



    shuffle(data)

    return data
pred_Images=data_making_for_prediction('../input/intel-image-classification/seg_pred/seg_pred')

pred_Images = np.array(pred_Images)
def get_classlabel(class_label):

    labels = {0:'buildings', 1:'forest',2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

    return labels[class_label]
from random import randint



f,ax = plt.subplots(2,2) 

f.subplots_adjust(0,0,3,3)

for i in range(0,2,1):

    for j in range(0,2,1):

        rnd_number = randint(0,len(pred_Images))

        ax[i,j].imshow(pred_Images[rnd_number])

        ax[i,j].set_title(get_classlabel(classifier.predict_classes(np.array(pred_Images[rnd_number]).reshape(-1,128,128,3))[0]))

        ax[i,j].axis('off')