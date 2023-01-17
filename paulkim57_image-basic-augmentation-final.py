!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import cv2

import pickle

from sklearn.model_selection import train_test_split

import os, os.path, shutil

import numpy as np

import random

from skimage import io 

from skimage.transform import rotate, AffineTransform, warp

from skimage import img_as_ubyte

from skimage.util import random_noise

def anticlockwise_rotation(image):

    angle= random.randint(0,180)

    return rotate(image, angle)



def clockwise_rotation(image):

    angle= random.randint(0,180)

    return rotate(image, -angle)



def h_flip(image):

    return  np.fliplr(image)



def v_flip(image):

    return np.flipud(image)



def add_noise(image):

    return random_noise(image)



def blur_image(image):

    return cv2.GaussianBlur(image, (9,9),0)



transformations = {'rotate anticlockwise': anticlockwise_rotation,

                      'rotate clockwise': clockwise_rotation,

                      'horizontal flip': h_flip, 

                      'vertical flip': v_flip,

                   'adding noise': add_noise,

                   'blurring image':blur_image

                 } 
images_path="../working/train/train" 

images=[] # to store paths of images from folder


for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     

    images.append(os.path.join(images_path,im))



    



images_to_generate=1300  #you can change this value according to your requirement

i=300                       # variable to iterate till images_to_generate



while i<=images_to_generate:    

    image=random.choice(images)

    original_image = io.imread(image)

    transformed_image=None

#     print(i)

    n = 0       #variable to iterate till number of transformation to apply

    transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image

    

    while n <= transformation_count:

        key = random.choice(list(transformations)) #randomly choosing method to call

        transformed_image = transformations[key](original_image)

        n = n + 1

    

    secondary_new_image_path = str(image[23:26] + str(i) + ".jpg") ### cv2.imwrite DOES NOT LIKE a "." except at the end

    new_image_path= images_path + "/" + secondary_new_image_path

    print(new_image_path)

    transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].

    transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it

    cv2.imwrite(new_image_path, transformed_image) # save transformed image to path



    i =i+1

    





import os



path, dirs, files = next(os.walk("../working/train/train"))

print(len(files))
### Sorting into different folders

path = "../working/train/train"

images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]



for image in images:

    folder_name = image[0:3]



    new_path = os.path.join(path, folder_name)

    if not os.path.exists(new_path):

        os.makedirs(new_path)



    old_image_path = os.path.join(path, image)

    new_image_path = os.path.join(new_path, image)

    shutil.move(old_image_path, new_image_path)
import os



path, dirs, files = next(os.walk("../working/train/train"))

file_count = len(files)

print(file_count)
path = '/kaggle/working/train/train'

catagories = os.listdir(path)

print(catagories)
X =[]

y = []



def create_data(img_size):

    for c in catagories:

        curr_path = os.path.join(path, c)

        classification = catagories.index(c)

        print(classification)

        for img in os.listdir(curr_path):

            try:

                img_array = cv2.imread(os.path.join(curr_path, img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (img_size, img_size))

                X.append(new_array)

                y.append(classification)

            except Exception as e:

                pass
create_data(100)
X = np.array(X).reshape(-1,100,100,1)

X = tf.keras.utils.normalize(X, axis = 1)

y = np.array(y)
unique, counts = np.unique(y, return_counts=True)

dict(zip(unique, counts))
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.1, random_state = 0)

unique, counts = np.unique(yTrain, return_counts=True)

dict(zip(unique, counts))
print(xTrain.shape[1:])
yTest = tf.keras.utils.to_categorical(yTest, 2)

yTrain = tf.keras.utils.to_categorical(yTrain, 2)
model = Sequential()



model.add(Conv2D(64, (5,5),input_shape = xTrain.shape[1:], activation = 'relu'))



model.add(MaxPooling2D(pool_size = (5,5)))



model.add(Flatten())

model.add(Dense(128, activation = 'sigmoid'))



model.add(Dense(2, activation="sigmoid"))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



model.fit(xTrain, yTrain, batch_size = 32, epochs = 3)
model.evaluate(xTest, yTest)
import sklearn.metrics as metrics





probs = model.predict_proba(xTest)

print(probs)



fpr, tpr, threshold = metrics.roc_curve(y_true = np.ravel(yTest), y_score = np.ravel(probs), pos_label = 1)

roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()