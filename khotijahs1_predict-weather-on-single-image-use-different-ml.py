import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import matplotlib.image as implt

from PIL import Image 

import seaborn as sns

import cv2 as cs2

import os



import warnings

warnings.filterwarnings('ignore')
## import Keras and its module for image processing and model building

import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
#copying the pretrained models to the cache directory

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)



#copy the Xception models

!cp ../input/keras-pretrained-models/xception* ~/.keras/models/

#show

!ls ~/.keras/models
train_path = "../input/twoclass-weather-classification/train"

test_path = "../input/twoclass-weather-classification/test"



train_cloudy = "/kaggle/input/twoclass-weather-classification/train/cloudy"

train_sunny = "/kaggle/input/twoclass-weather-classification/train/sunny"



test_cloudy = "/kaggle/input/twoclass-weather-classification/test/cloudy"

test_sunny = "/kaggle/input/twoclass-weather-classification/test/sunny"
# VISUALIZATION

category_names = os.listdir(train_path) # output: ['sunny', 'cloudy']

nb_categories = len(category_names) # output: 2

train_images = []



for category in category_names:

    folder = train_path + "/" + category

    train_images.append(len(os.listdir(folder)))



sns.barplot(y=category_names, x=train_images).set_title("Number Of Training Images Per Category");
img = load_img('../input/twoclass-weather-classification/train/cloudy/c0001.jpg')  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array 

print('image shape: ', x.shape)



print('Train Cloudy Image')

plt.imshow(img)

plt.show()





img = load_img('../input/twoclass-weather-classification/train/sunny/s0001.jpg')  # this is a PIL image

x = img_to_array(img)  # this is a Numpy array 

print('Train Sunny Image')

plt.imshow(img)

plt.show()



img_size = 50

cloudy_train = []

sunny_train = []

label = []



for i in os.listdir(train_cloudy): # all train cloudy images

    if os.path.isfile(train_path + "/cloudy/" + i): # check image in file

        cloudy = Image.open(train_path + "/cloudy/" + i).convert("L") # converting grey scale 

        cloudy = cloudy.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50

        cloudy = np.asarray(cloudy)/255 # bit format

        cloudy_train.append(cloudy)

        label.append(1)

        

for i in os.listdir(train_sunny): # all train sunny images

    if os.path.isfile(train_path + "/sunny/" + i): # check image in file

        sunny = Image.open(train_path + "/sunny/" + i).convert("L") # converting grey scale 

        sunny = sunny.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50

        sunny = np.asarray(sunny)/255 # bit format

        sunny_train.append(sunny)

        label.append(0)
x_train = np.concatenate((cloudy_train,sunny_train),axis=0) # training dataset

x_train_label = np.asarray(label) # label array containing 0 and 1

x_train_label = x_train_label.reshape(x_train_label.shape[0],1)



print("cloudy:",np.shape(cloudy_train) , "sunny:",np.shape(sunny_train))

print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
# Visualizing Training data

print(x_train_label[0])

plt.imshow(cloudy_train[0])
# Visualizing Training data

print(x_train_label[0])

plt.imshow(sunny_train[0])
img_size = 50

cloudy_test = []

sunny_test = []

label = []





for i in os.listdir(test_cloudy): # all train cloudy images

    if os.path.isfile(test_path + "/cloudy/" + i): # check image in file

        cloudy = Image.open(test_path + "/cloudy/" + i).convert("L") # converting grey scale 

        cloudy = cloudy.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50

        cloudy = np.asarray(cloudy)/255 # bit format

        cloudy_test.append(cloudy)

        label.append(1)

        

for i in os.listdir(test_sunny): # all train sunny images

    if os.path.isfile(test_path + "/sunny/" + i): # check image in file

        sunny = Image.open(test_path + "/sunny/" + i).convert("L") # converting grey scale 

        sunny = sunny.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50

        sunny = np.asarray(sunny)/255 # bit format

        sunny_test.append(sunny)

        label.append(0)
x_test = np.concatenate((cloudy_test,sunny_test),axis=0) # training dataset

x_test_label = np.asarray(label) # label array containing 0 and 1

x_test_label = x_test_label.reshape(x_test_label.shape[0],1)



print("cloudy:",np.shape(cloudy_test) , "sunny:",np.shape(sunny_test))

print("test_dataset:",np.shape(x_test), "test_values:",np.shape(x_test_label))
# Visualizing Training data

print(x_test_label[0])

plt.imshow(cloudy_test[0])
# Visualizing Training data

print(x_test_label[0])

plt.imshow(sunny_test[0])
x = np.concatenate((x_train,x_test),axis=0) # count: train_data

# x.shape: 

#   output 

y = np.concatenate((x_train_label,x_test_label),axis=0) # count: test_data

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flatten 3D image array to 2D, count: 50*50 = 2500

print("images:",np.shape(x), "labels:",np.shape(y))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]



print("Train Number: ", number_of_train)

print("Test Number: ", number_of_test)
x_train = X_train.T

x_test = X_test.T

y_train = y_train.T

y_test = y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

test_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
# with GridSearchCV

from sklearn.model_selection import GridSearchCV



grid = {

    "C": np.logspace(-4, 4, 20),

    "penalty": ["l1","l2"]

}

lg=LogisticRegression(random_state=42)

log_reg_cv=GridSearchCV(lg,grid,cv=10,n_jobs=-1,verbose=2)

log_reg_cv.fit(x_train.T,y_train.T)

print("accuracy: ", log_reg_cv.best_score_)
models = pd.DataFrame({

    'Model': ['LR with sklearn','LR with GridSearchCV' ],

    'Train Score': [train_acc_logregsk, "-"],

    'Test Score': [test_acc_logregsk, log_reg_cv.best_score_*100]

})

models.sort_values(by='Test Score', ascending=False)
from sklearn.ensemble import BaggingClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



# We define the SVM model

svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=5,kernel='rbf',random_state=42, probability=True), 

                                               n_jobs=-1))

test_acc_svm = round(svmcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_svm = round(svmcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

model2 = pd.DataFrame({

    'Model': ['SVM'],

    'Train Score': [train_acc_svm],

    'Test Score': [test_acc_svm*100]

})

model2.sort_values(by='Test Score', ascending=False)
from sklearn.ensemble import RandomForestClassifier



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)

test_acc_rfcla = round(rfcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_rfcla = round(rfcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
model3 = pd.DataFrame({

    'Model': ['Random Forest'],

    'Train Score': [train_acc_rfcla],

    'Test Score': [test_acc_rfcla*100]

})

model3.sort_values(by='Test Score', ascending=False)
from sklearn.tree import DecisionTreeClassifier



# We define the model

dtcla =  DecisionTreeClassifier(random_state=9)

test_acc_dtcla = round(dtcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_dtcla = round(dtcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
model4 = pd.DataFrame({

    'Model': ['Decision Tree'],

    'Train Score': [train_acc_dtcla],

    'Test Score': [test_acc_dtcla*100]

})

model4.sort_values(by='Test Score', ascending=False)
model5 = pd.DataFrame({

    'Model': ['train_acc_logregsk', 'train_acc_svm','train_acc_rfcla','Decision Tree'],

    'Train Score': [train_acc_logregsk, train_acc_svm , train_acc_rfcla ,train_acc_dtcla],

    'Test Score': [test_acc_logregsk, test_acc_svm , test_acc_rfcla ,test_acc_dtcla*100]

})

model5.sort_values(by='Test Score', ascending=False)