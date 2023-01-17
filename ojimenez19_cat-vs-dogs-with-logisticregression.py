import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

import cv2

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



training_images_folder = '/train/train'

test_images_folder = '/test/test1'



image_size = 32



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as train_zip:

    train_zip.extractall('/train')

    

with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as test_zip:

    test_zip.extractall('/test')



print('Files extracted')

print("Training files:",len(os.listdir(training_images_folder)))

print("Test files:",len(os.listdir(test_images_folder)))
# Read categories to get the y array

train_files = os.listdir(training_images_folder)

categories = []

for img in train_files: 

    categories.append(img.split('.')[0])

y = np.array(categories)

print(y)
def get_image_arr(path):

        img = cv2.imread(path, cv2.COLOR_BGR2GRAY) 

        img = cv2.resize(img, (image_size, image_size)).reshape(1,-1)

        return img
def get_features_count(path):

        img = cv2.imread(path, cv2.COLOR_BGR2GRAY) 

        img = cv2.resize(img, (image_size, image_size)).reshape(-1,1)

        return len(img)
features_count = get_features_count(training_images_folder + '/' + train_files[0])

print('Features count: ',features_count)
#Create the X array

X = np.ones((1,features_count), int)

advance = 0



for x in train_files:

    X = np.append(X,get_image_arr(training_images_folder + '/' + x),axis = 0)



print('X traing array generated',X)
X = np.delete(X, (0), axis=0)



print('X shape:',X.shape)

print('y shape:',y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



print('X Train shape:',X_train.shape)

print('X Test shape:',X_test.shape)

print('Y Train shape:',y_train.shape)

print('Y Test shape:',y_test.shape)
#Scale data

train_scaler = preprocessing.RobustScaler().fit(X_train)

x_scaled = train_scaler.transform(X_train)



test_scaler = preprocessing.RobustScaler().fit(X_test)

test_scaled = test_scaler.transform(X_test)
images_model = LogisticRegression(max_iter=2000)

images_model.fit(x_scaled, y_train)



y_prima = images_model.predict(test_scaled)

print(y_prima)
#Statistics 

print(classification_report(y_test, y_prima))
#Accuracy

accuracy_score(y_test, y_prima)
#confusion matrix

disp = plot_confusion_matrix(images_model, X_train, y_train,

                             cmap=plt.cm.Blues,

                             normalize=None)

plt.show()