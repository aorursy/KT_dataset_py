# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from zipfile import ZipFile

import cv2

#Extracting zip files 

zf = ZipFile('../input/dogs-vs-cats/train.zip', 'r')

zf1 = ZipFile('../input/dogs-vs-cats/test1.zip', 'r')

zf.extractall('../kaggle/working/Temp')

zf.close()

zf1.extractall('../kaggle/working/Temp')

zf1.close()

main = "../kaggle/working/Temp/"

train_dir="train"

p=os.path.join(main,train_dir)

p

os.listdir(p)

example_pic='dog.890.jpg'

path_example=p+'/'+example_pic

X=[]

Y=[]

convert = lambda category : int(category == 'dog')  #Function to convert the dogs name as 0 and cat as 1

for file in os.listdir(p):

    category=file.split('.')[0]

    category=convert(category)

        

    img_array=cv2.imread(os.path.join(p,file))

    new_array = cv2.resize(img_array, dsize=(80, 80))

    X.append(new_array)

    Y.append(category)

Y

X=X[:10000] #Taking 10000 images as training set

Y=Y[:10000]

Y

X=np.array(X).reshape(-1,80,80,3) ##reshaping the array with image_width,image_size and channels to provide in conv2d

Y=np.array(Y)

Y.shape

X.shape

X=X/255

#Printing an example Image

from keras.preprocessing.image import load_img,ImageDataGenerator,img_to_array

img=load_img(path_example)

plt.imshow(img)

from keras.utils import to_categorical

Y=to_categorical(Y)

Y

Y.shape

from keras.layers import Dense, Conv2D,Dropout,Flatten,MaxPool2D,GlobalAveragePooling2D

from keras.models import Sequential,Model

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

#Using pretraine Inception v3 model

from keras.applications.inception_v3 import InceptionV3, preprocess_input

inc_model = InceptionV3(weights='imagenet',

                        include_top=False,

                        input_shape=(80,80,3))

                       



print("number of layers:", len(inc_model.layers))

inc_model.summary()

#Adding custom layers

x = inc_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(64, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)

model_ = Model(inputs=inc_model.input, outputs=predictions)

#Lock initial layers to not be trained

for layer in model_.layers[:52]:

    layer.trainable=False

model_.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_.fit(X,Y,validation_split=0.2,batch_size=32,epochs=30,callbacks=[early_stopping])

#Preprocessing test set

main = "../kaggle/working/Temp/"

test_dir="test1"

p=os.path.join(main,test_dir)

p

test_filename=os.listdir("../kaggle/working/Temp/test1")

test_filename

x_test=[]

name=[]

for path in os.listdir(p):

    name.append(path.split(".")[0])

    img_array = cv2.imread(os.path.join(p,path))

    new_img_array = cv2.resize(img_array, dsize=(80, 80))

    x_test.append(new_img_array)

len(x_test)    

x_test = np.array(x_test)

x_test.shape

x_test=x_test.reshape(-1,80,80,3)

x_test.shape

x_test = x_test/255

x_test

pred=model_.predict(x_test)

#Showing first 5 test images

f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

plt.title('Actual testing set')

for i in range(5):

    ax[i].imshow(x_test[i])

plt.show()



predicted_val = [int(round(p[0])) for p in pred]

predicted_val

#Submission Dataframe

submission_df = pd.DataFrame({'id':name, 'label':predicted_val})

submission_df.head(10)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session