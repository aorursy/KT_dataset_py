import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_labels_df = pd.read_csv("../input/train_labels.csv")

train_labels_df.head()
train_labels_df.tail()
train_labels_df.describe() #describing train_labels_df
train_labels_df.invasive.value_counts() #finding how many invasive and not invasive samples in train data
#Getting image names from both train and test folders

train_images_names = check_output(["ls", "../input/train/"]).decode("utf8")

train_images_names = train_images_names.split("\n")

test_images_names = check_output(["ls", "../input/test/"]).decode("utf8")

test_images_names = test_images_names.split("\n")

print("Total train images",len(train_images_names))

print("Total test images",len(test_images_names))
test_df = pd.DataFrame()

test_df["name"] = [test_image.split(".")[0] for test_image in test_images_names]

test_df.head()
% pylab inline

import os

import random



import pandas as pd

from scipy.misc import imread

print("See train images with invasive and without invasive species")

print("3.jpg - With Invasive species")

img = imread("../input/train/3.jpg")

imshow(img)
print("4.jpg - Non-Invasive species")

img1 = imread("../input/train/4.jpg")

imshow(img1)
#importing all the necessary modules

% pylab inline

import os

import random



import pandas as pd

from scipy.misc import imread



root_dir = os.path.abspath('.')

data_dir = '../input/'
i = random.choice(train_labels_df.index)



img_name = str(train_labels_df.name[i])+".jpg"

img = imread(os.path.join(data_dir, 'train', img_name))



imshow(img)

print("Image",img_name)

print("Invasive", train_labels_df.invasive[i])
#Resizing train images

from scipy.misc import imresize



temp = []





for img_name in train_labels_df.name:

    img_path = os.path.join(data_dir, 'train', str(img_name)+".jpg")

    img = imread(img_path)

    img = imresize(img, (32, 32))



    img = img.astype('float32')

    temp.append(img)

train_x = np.stack(temp)
print(test_df.tail()) #Last row is null

test_df = test_df[:-1] #So removing last row from the test dataframe

print(test_df.tail())
#Resizing test images

temp = []

i=0

for img_name in test_df.name:

    img_path = os.path.join(data_dir, 'test', str(img_name)+".jpg")

    try:

        img = imread(img_path)

        img = imresize(img, (32, 32))



        img = img.astype('float32')

        temp.append(img)

        i=i+1

    except:

        continue

test_x = np.stack(temp)
train_x = train_x / 255

test_x = test_x / 255
train_labels_df.invasive.value_counts(normalize=True)
import keras

from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

train_y = lb.fit_transform(train_labels_df.invasive)

train_y = keras.utils.np_utils.to_categorical(train_y)



input_num_units = (32, 32, 3)

hidden_num_units = 500

output_num_units = 2



epochs = 5

batch_size = 128
#Import the necessary keras modules

from keras.models import Sequential

from keras.layers import Dense, Flatten, InputLayer



#Define our network

model = Sequential([

  InputLayer(input_shape=input_num_units),

  Flatten(),

  Dense(units=hidden_num_units, activation='relu'),

  Dense(units=output_num_units, activation='softmax'),

])
model.summary()
#Compile and train our network

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1)
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)
pred = model.predict_classes(test_x)

pred = lb.inverse_transform(pred)

test_df['invasive'] = pred
test_df['invasive'].value_counts()
test_df.to_csv('submission.csv', index=False)
i = random.choice(train_labels_df.index)

img_name = train_labels_df.name[i]



img = imread(os.path.join(data_dir, 'train', str(img_name)+".jpg")).astype('float32')

imshow(imresize(img, (128, 128)))

pred = model.predict_classes(train_x)

print('Original:', train_labels_df.invasive[i], 'Predicted:', lb.inverse_transform(pred[i]))