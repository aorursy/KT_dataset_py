# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
seed = 42
rng = np.random.RandomState(seed)
data = pd.read_csv("../input/facial_keypoint_identification/facial_keypoint_identification/facial_keypoint_identification.csv")
data.head()
# get random index of data
idx = rng.choice(data.index)
img = plt.imread("../input/facial_keypoint_identification/facial_keypoint_identification/images/" + data["image_name"].iloc[idx])
fig = plt.figure()

# plot the image
plt.imshow(img)

# plot the target
plt.scatter(data["left_eye_center_x"].iloc[idx],data["left_eye_center_y"].iloc[idx],marker="x",c="r")
plt.scatter(data["right_eye_center_x"].iloc[idx],data["right_eye_center_y"].iloc[idx],marker="x",c="r")

# print the shape of the image
print("Shape of the image is",img.shape)

# show figure
plt.show()
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, VGG19
from sklearn.model_selection import train_test_split
images = []
for img_name in data["image_name"]:
    img = image.load_img("../input/facial_keypoint_identification/facial_keypoint_identification/images/"+img_name,target_size=(224,224,3))
    img = image.img_to_array(img)
    images.append(img)

images = np.array(images)
images.shape
X = preprocess_input(images,mode="tf")
y = data.iloc[:,1:].values
y.shape
X_train,X_valid,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from keras.models import Sequential
from keras.layers import Dense,Dropout,InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
features_train = base_model.predict(X_train)
features_valid = base_model.predict(X_valid)
max_val = features_train.max()
features_train /= max_val
features_valid /= max_val
features_train = features_train.reshape(features_train.shape[0], 7*7*512)
features_valid = features_valid.reshape(features_valid.shape[0], 7*7*512)
model=Sequential()
model.add(InputLayer((7*7*512, )))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=4, activation='linear'))

adam = Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer=adam)
estop = EarlyStopping(patience=10, mode='min', min_delta=0.001, monitor='val_loss')
model.fit(features_train, Y_train, epochs=200, batch_size=512, callbacks=[estop] ,validation_data=(features_valid,Y_test))
predictions = model.predict(features_valid)
predictions
_, valid_images, _, _ = train_test_split(data.image_name,y,test_size=0.3, random_state=42)
idx = rng.choice(range(len(valid_images)))
image_name = valid_images.iloc[idx]
left_eye_center_x = data.loc[data.image_name == image_name, 'left_eye_center_x']
left_eye_center_y = data.loc[data.image_name == image_name, 'left_eye_center_y']
right_eye_center_x = data.loc[data.image_name == image_name, 'right_eye_center_x']
right_eye_center_y = data.loc[data.image_name == image_name, 'right_eye_center_y']
predicted_left_eye_center_x, predicted_left_eye_center_y, predicted_right_eye_center_x, predicted_right_eye_center_y = predictions[idx]
img = plt.imread('../input/facial_keypoint_identification/facial_keypoint_identification/images/' + image_name)

# plot empty figure
fig = plt.figure()

# plot image
plt.imshow(img,cmap='gray')

# plot actual targets
plt.scatter(left_eye_center_x, left_eye_center_y, marker='x', s=50, c='r')
plt.scatter(right_eye_center_x, right_eye_center_y, marker='x', s=50, c='r')

# plot predictions
plt.scatter(predicted_left_eye_center_x, predicted_left_eye_center_y, marker='o', s=50, c='w')
plt.scatter(predicted_right_eye_center_x, predicted_right_eye_center_y, marker='o', s=50, c='w')

# show the figure
plt.show()
