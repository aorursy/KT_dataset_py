# Libraries

# linear algebra and data manipulation

import numpy as np

# For plotting faces

import matplotlib.pyplot as plt

from skimage.io import imshow

# Regresseros

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import GradientBoostingRegressor
# Input data files consisting of the images

img_ds = np.load("../input/olivetti_faces.npy")

img_ds.shape
# Sample images of a subject

img_cnt = 10

plt.figure(figsize=(12,12))

for i in range(img_cnt):

    plt.subplot(1,10,i+1)

    x=img_ds[i+30] # 3rd subject

    imshow(x)

plt.show()
targets = np.load("../input/olivetti_faces_target.npy")

targets.shape
# Flatten image data set to 2 dimensional array

img_ds_flt = img_ds.reshape(img_ds.shape[0],img_ds.shape[1]*img_ds.shape[2])
# Patition dataset into training and test groups (fancy indexing)

targets < 30                # Output is true/false

train = img_ds_flt[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = img_ds_flt[targets >= 30]  # Test on rest independent people  10 * 10 = 100
# Generate 8 random integers between 0 and 100

n_faces = test.shape[0]//12             # // is unconditionally "flooring division"

face_ids = np.random.randint(0 , 100, size = n_faces)

# Random 'n_faces' faces from within 1 to 100

sub_test = test[face_ids, :]

face_ids
# Total pixels in any image

n_pixels = img_ds_flt.shape[1]

# Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]    # // "flooring division", 3.1//1.2 = 2.0

# Lower half of the faces will be target(s)

y_train = train[:, n_pixels // 2:]

# Similarly for test data. Upper and lower half

X_test = sub_test[:, :(n_pixels + 1) // 2]

y_test = sub_test[:, n_pixels // 2:]
# Prepare a dictionary of estimators after instantiating each one of them

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,     # Out of 20000

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),                          # Accept default parameters

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "multi_gbm" : MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))

}

# Create an empty dictionary to collect prediction values

y_test_predict = dict()
# Fit each model by turn and make predictions

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

                                                       # Note that output of estimator.predict(X_test) is prediction for

                                                       #  all the test images and NOT one (or one-by-one)



## Processing output -> Each face should have this dimension

image_shape = (64, 64)
plt.figure(figsize=(10,10))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face.reshape(image_shape)

    imshow(y)

  

plt.show()
plt.figure(figsize=(10,10))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face.reshape(image_shape)

    imshow(y)

  

plt.show()
plt.figure(figsize=(15,15))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face

    imshow(y)

  

plt.show()
plt.figure(figsize=(10,10))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face.reshape(image_shape)

    imshow(y)

  

plt.show()
plt.figure(figsize=(10,10))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['multi_gbm'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face.reshape(image_shape)

    imshow(y)

  

plt.show()