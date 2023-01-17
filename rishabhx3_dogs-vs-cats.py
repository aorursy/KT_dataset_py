import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator, load_img



from PIL import Image

import cv2

from zipfile import ZipFile

import os
file_train = "../input/dogs-vs-cats/train.zip"

file_test = "../input/dogs-vs-cats/test1.zip"



with ZipFile(file_train, 'r') as zip:

    zip.extractall('/train')

    print('Train Extract Done!') 

    

with ZipFile(file_test,'r') as zip:

    zip.extractall('/test1')

    print('Test Extract Done!')
print('Train Data ',len(os.listdir('/train/train/')))

print('Test Data ',len(os.listdir('/test1/test1/')))
rand_pic = np.random.randint(0,len(os.listdir('/train/train/')))

dc_pic = os.listdir('/train/train/')[rand_pic]



# Load the images

dc_load = Image.open('/train/train/' + dc_pic)

category = dc_pic.split(".")[0]

plt.title(category)

img_plot = plt.imshow(dc_load)
train_path = '/train/train/'



X_train = []

y_train = []



convert = lambda category : int(category == 'dog')



def create_train_data(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category = convert(category)

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X_train.append(new_img_array)

        y_train.append(category)

        

create_train_data(train_path)



X_train = np.array(X_train).reshape(-1, 80,80,1)

y_train = np.array(y_train)

X_train = X_train/255.0
test_path = "/test1/test1/"



X_test = []

test_id = []



def create_test_data(path):

    for p in os.listdir(path):

        test_id.append(p.split(".")[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X_test.append(new_img_array)



create_test_data(test_path)



X_test = np.array(X_test).reshape(-1,80,80,1)

X_test = X_test/255
model = Sequential()



model.add(Conv2D(16,(3,3), activation = 'relu', input_shape = X_train.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(32,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(128,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.3))

model.add(Flatten())



model.add(Dense(512, activation='relu'))



model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 20, batch_size = 100, validation_split=0.3)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and Validation accuracy')

plt.legend(loc=0)

plt.figure()



plt.show()
predictions = model.predict(X_test)

predicted_val = [int(round(p[0])) for p in predictions]
submission_df = pd.DataFrame({'id':test_id, 'label':predicted_val})

submission_df.to_csv("submission.csv", index=False)
sample_test = submission_df.head(60)

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['id']

    category = row['label']

    img = load_img("/test1/test1/"+filename+".jpg", target_size=(128,128))

    plt.subplot(10, 6, index+1)

    plt.imshow(img)

    if(category == 1):

        plt.title( '(' + "Dog"+ ')' )

    else:

        plt.title( '(' + "Cat"+ ')' )

plt.tight_layout()

plt.show()