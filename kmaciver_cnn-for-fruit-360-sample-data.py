import matplotlib.pyplot as plt 

from PIL import Image 

import seaborn as sns

import pandas as pd 

import numpy as np 

import os



# Get the categories for each label



data_path = "../input/sample-fruit-set/sample_training/sample_training"

CATEGORIES = []



for filename in os.listdir(data_path):

    CATEGORIES.append(filename)

print(CATEGORIES)       

#Number of different fruits

len(CATEGORIES)
# Loading Training data

from keras.preprocessing import image



training_data=[]

n_categories=[]



def create_training_data():

    for category in CATEGORIES:

        folder_path = os.path.join(data_path, category).replace("\\","/")

        class_num = CATEGORIES.index(category)

        counter = 0

        for i in os.listdir(folder_path):

            try:

                img_path = os.path.join(folder_path, i).replace("\\","/")

                img = image.load_img(img_path, target_size=(224, 224))

                img = image.img_to_array(img)

                training_data.append([img, class_num])

                counter += 1

            except Exception as e:

                pass

        n_categories.append(counter)

create_training_data()
%matplotlib inline 

plt.figure(figsize=(10,10))

plt.bar(CATEGORIES, n_categories)

plt.xticks(CATEGORIES, rotation=30)
print(n_categories)
sum(n_categories)
end_tail=0

for i in range(12,3,-1):

   end_tail+= n_categories[i]



front_slice=490

for i in range(3):

    front_slice+=n_categories[i]



training_front_slice = training_data[:front_slice]

training_end_slice =  training_data[-end_tail:]



training_data = training_front_slice + training_end_slice
len(training_data)
from itertools import groupby



cat=[]

for i in training_data:

    cat.append(i[1])    

n_categories_balance= [len(list(group)) for key, group in groupby(cat)]



plt.figure(figsize=(10,10))

plt.bar(CATEGORIES, n_categories_balance)

plt.xticks(CATEGORIES, rotation=30)
# Now that data is balanced, it is time to shuffle training data.

import random

random.shuffle(training_data)
# Preparing training data for CNN

X_train = []

Y_train = []

for img, class_num in training_data:

    X_train.append(img)

    Y_train.append(class_num)



X_train = np.array(X_train).reshape(-1, 224, 224, 3)

Y_train = np.array(Y_train)

X_train.shape
# Generate Test data

testing_data = []

n_test_categories = []

test_path = "../input/sample-fruit-set/sample_test/sample_test"

test_CATEGORIES = []



for filename in os.listdir(test_path):

    test_CATEGORIES.append(filename)      



def create_test_data():

    for category in test_CATEGORIES:

        folder_path = os.path.join(test_path, category).replace("\\","/")

        class_num = test_CATEGORIES.index(category)

        counter = 0

        for i in os.listdir(folder_path):

            try:

                img_path = os.path.join(folder_path, i).replace("\\","/")

                img = image.load_img(img_path, target_size=(224, 224))

                img = image.img_to_array(img)

                testing_data.append([img, class_num])

                counter += 1

            except Exception as e:

                pass

        n_test_categories.append(counter)

create_test_data()



plt.figure(figsize=(10,10))

plt.bar(test_CATEGORIES, n_test_categories)

plt.xticks(test_CATEGORIES, rotation=30)
# Prepare test data

random.shuffle(testing_data)

X_test = []

Y_test = []

for img, class_num in testing_data:

    X_test.append(img)

    Y_test.append(class_num)



X_test = np.array(X_test).reshape(-1, 224, 224, 3)

X_test.shape
len(n_test_categories)
from keras import utils

Y_train = utils.to_categorical(Y_train, num_classes=len(n_categories))

Y_test = utils.to_categorical(Y_test, num_classes=len(n_test_categories))
# Normalization of data for CNN

X_train_ = X_train/255

X_test_ = X_test/255
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint





def baseline_model():

    model=Sequential()

    model.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu" , input_shape=(224,224,3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"))

    #model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"))

    #model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

    #model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))

    model.add(Dense(13, activation='softmax'))



    sgd = SGD(lr=0.01)

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)



    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model





model = baseline_model()

es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.01)

mc = ModelCheckpoint("best_fruit_classifier.h5", monitor="val_acc", mode="max", save_best_only=True)

history = model.fit(X_train_, Y_train, validation_data=(X_test_, Y_test),  batch_size=10, epochs=150, callbacks=[es, mc])



# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from keras.models import load_model

fruitmodel = load_model("best_fruit_classifier.h5")

fruitmodel.summary()
from PIL import Image

import seaborn as sns



#Create a new sample data

sample_data = testing_data

random.shuffle(sample_data)

sample_data = sample_data[0:4]



X_sample = []

Y_sample = []



for img, class_num in sample_data:

    X_sample.append(img)

    Y_sample.append(class_num)



# Prepare sample data for prediction

X_sample_array = np.array(X_sample).reshape(-1, 224, 224, 3)

X_sample_array_ = X_sample_array/255 



#Plot images of sample data

f, ax = plt.subplots(1, 4)

f.set_size_inches(20,20)

for i in range(4):

    img = Image.fromarray(X_sample_array[i].astype('uint8'))

    ax[i].imshow(img)

    ax[i].set_title(CATEGORIES[Y_sample[i]])

plt.show()



# Plot the predictions made for the sample data

f, axes = plt.subplots(1, 4)

f.set_size_inches(80,20)

preds  = fruitmodel.predict(X_sample_array_)

for i in range(len(preds)):    

    arr= np.array(preds[i])

    arr = arr.argsort()[-3:][::-1]

    xplot = [preds[i][c] for c in arr]

    yplot = [CATEGORIES[c] for c in arr]

    b = sns.barplot(y=yplot, x=xplot, color="gray", ax=axes[i])

    b.tick_params(labelsize=55)

    f.tight_layout()