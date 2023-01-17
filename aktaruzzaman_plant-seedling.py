#Import all general libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

import cv2

import glob





%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
#This function will return all training image paths and corrosponding lebels were collected in Series object



def train_impath():

    impaths = [f for f in glob.glob("../input/train/*")]

    label = pd.Series(impaths).str.split('/').apply(lambda x:x[-1])



    imagePaths = []

    for lab in label:

        imp = [f for f in glob.glob("../input/train/"+lab+"/*.png")]

        imagePaths.extend(imp)

    imagePaths = pd.Series(imagePaths)

    labels = pd.Series(imagePaths).str.split('/').apply(lambda x:x[3])

    labels = pd.Series(labels)

    return labels, imagePaths
impaths = [f for f in glob.glob("../input/train/*")]

pd.Series(impaths)
#This function will return all training image paths were collected in Series object

#No labels. we will find them. thats our challenge

def test_impath():

    labels = pd.read_csv('../input/sample_submission.csv')

    lab = labels.iloc[:,0].tolist()



    data_dir = []

    for l in lab:

        t_dir = "../input/test/"+l

        data_dir.append(t_dir)

    return data_dir
#Calling both functions above and get lebels for training data set and training and test image PATHS

#Note: we will encode all the labels later by using onehot encoding

labels, train_impaths = train_impath()

test_impaths = test_impath()
#Now all image paths and lebels are taken. Using keras preprocessing tools we are going to read all images in those paths

#This function will read image and return them as ndarray(). Before return images will be normalized by deviding 255. Normalaizing will help to reduce the number value and will save computational cost

#image load by using keras model

from keras.preprocessing import image

def im_read(impaths):

    df = []

    for path_ in impaths:

        img = image.load_img(path=path_,target_size=(224,224,3))

        img = image.img_to_array(img)

        df.append(img)

    return np.array(df, dtype=float)/255
#Calling keras image read function and load train and test iamges

X = im_read(train_impaths)

X_ts = im_read(test_impaths)
#Sample image show

plt.imshow(X[0])
#labels were loaded few cells before and were mentioned that we need to encode them by using onehot encoding

#All the lebels need to be encoded as a matrix/ndarray. For instance if label is '1' and total number of calss is 3. '1' need to be encoded as [1,0,0] for the first image/observation as lebeled '1'

#Sklearn OneHotEncoder and LebelEncoder is a very good tools to encode labels

#In this problem we are using LabelEncoder



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

le = LabelEncoder()



labels_encod = labels.value_counts().index.sort_values()



le = le.fit(labels_encod)

labels = le.transform(labels)

labels
#All the lebels are converted in numberse. Now we need to do oneHot encoding which will convert them as ndarray

from keras.utils import to_categorical as tc

Y = tc(labels,num_classes=12)
#Model has been designed using transfer learning approach. VGG16 model was used and layers from input to 'fc2' were take and a custom layer was added as it requires to fit our problem.



from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import decode_predictions

from keras.models import Model

from keras.layers import Input, Dense



input_ = Input(shape=(224, 224, 3))



model = VGG16(input_tensor=input_)

model.summary()



number_of_class=12

vgg16_fc2 = model.get_layer('fc2').output

My_out_layer = Dense(number_of_class, activation='softmax',name = 'custome_layer1')(vgg16_fc2)



My_model = Model(input_,My_out_layer)

My_model.summary()
#All VGG16 layers are set as non-trainable and only custom layer will be open to train

for layer in My_model.layers[:-1]:

    layer.trainable = False

My_model.summary()
#SGD optimization was used as a suitable optimization technique for this proble.

from keras import optimizers

opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

My_model.compile(optimizer = opt , loss = 'mse', metrics=['mae','accuracy'])
#Fit model: Here all the input parameter(images) will be fit with there corresponding labels

My_model.fit(X, Y, epochs=1, batch_size=30,validation_split = 0.2)
#Predict labels from test images. It will return an array of float values for each test image. That array will contain probability for each catagory

Y_ts = My_model.predict(X_ts)
# select the indix with the maximum probability which will be assigned as the label for each of the images

results = np.argmax(Y_ts,axis = 1)



results = le.inverse_transform(results)

results = pd.Series(results, name='species')
labels = pd.read_csv('../input/sample_submission.csv')

lab = labels.iloc[:,0].tolist()
file = pd.Series(lab,name="file")
sub = pd.concat([pd.DataFrame(file),pd.DataFrame(results)],axis=1, sort=False)
sub.to_csv('sub11.csv',index=False)