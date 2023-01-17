#Install Pydrive
!pip install PyDrive
#Import necessary Libraries
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
#Creating a drive variable to access Google Drive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
download = drive.CreateFile({'id': '1_W2gFFZmy6ZyC8TPlxB49eDFswdBsQqo'})
download.GetContentFile('face_mask_detection.zip')
!unzip face_mask_detection.zip
#Import Libraries for model building Phase
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
train = pd.read_csv('face_mask_detection/Training_set_face_mask.csv')
labels = pd.read_csv('face_mask_detection/Training_set_face_mask.csv')   # loading the labels
labels.head()
file_paths = [[fname, 'face_mask_detection/train/' + fname] for fname in labels['filename']]
# Confirm if number of images is same as number of labels given
if len(labels) == len(file_paths):
    print('Number of labels i.e. ', len(labels), 'matches the number of filenames i.e. ', len(file_paths))
else:
    print('Number of labels does not match the number of filenames')
images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])
images.head()
train_data = pd.merge(images, labels, how = 'inner', on = 'filename')
train_data.head()   
train_data.shape[0]
train_data['filepaths'][1]
#Using OpenCV library for Image manipulation
import cv2
img = cv2.imread(train_data['filepaths'][1])
print(img.shape)
plt.imshow(img)
train_image = []
for i in tqdm(range(train_data.shape[0])):
    img = image.load_img('face_mask_detection/'+'train/' + train_data['filename'][i], target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
#Encoding the target variable as Integers
class_list = train_data['label'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(class_list))}
y_train = [Y_train[k] for k in class_list]
#One-hot encoding the target variable
from keras.utils import to_categorical
y = to_categorical(y_train)
#Creating a validation set from the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
y_test.shape
#Defining the Model Structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
#Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#Training the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# Loading the order of the image's name that has been provided
test_image_order = pd.read_csv("face_mask_detection/Testing_set_face_mask.csv")
test_image_order.head()
file_paths = [[fname, 'face_mask_detection/test/' + fname] for fname in test_image_order['filename']]
# Confirm if number of images is same as number of labels given
if len(test_image_order) == len(file_paths):
    print('Number of image names i.e. ', len(test_image_order), 'matches the number of file paths i.e. ', len(file_paths))
else:
    print('Number of image names does not match the number of filepaths')

test_images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])
test_images.head()
test_data = pd.merge(test_images, test_image_order, how = 'inner', on = 'filename')
test_data.head()  
test_data.drop('label', axis = 1, inplace=True)
import cv2
img = cv2.imread(test_data['filepaths'][1])
print(img.shape)
plt.imshow(img)
test_image = []
for i in tqdm(range(test_data.shape[0])):
    img = image.load_img('face_mask_detection/'+'test/' + test_data['filename'][i], target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
pred = model.predict_classes(test)
pred[0]
np.round(pred[0])
res = pd.DataFrame({'filename': test_data['filename'], 'label': pred})  # prediction is nothing but the final predictions of your model on input features of your new unseen test data
res.to_csv("submission.csv") 

# To download the csv file locally
from google.colab import files        
files.download('submission.csv')
