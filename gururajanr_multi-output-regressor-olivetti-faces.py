# Call libraries

# For data manipulation

import numpy as np



# For plotting faces

import matplotlib.pyplot as plt   

from skimage.io import imshow



# Regressors

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor
data_images = np.load("../input/olivetti_faces.npy")

data_images.shape
data_targets = np.load("../input/olivetti_faces_target.npy")

data_targets.shape   
# See an image

firstImage = data_images[0]

imshow(firstImage) 
# Flatten each image

data = data_images.reshape(data_images.shape[0], data_images.shape[1] * data_images.shape[2])     # 64 X 64 = 4096

# Flattened 64 X 64 array

data.shape  
# Partition datasets into two (fancy indexing)

data_targets < 30                # Output is true/false
 # First 30 types of images out of 40 ie 30 * 10 =300

train = data[data_targets < 30] 

train.shape
 # Test on rest independent people  10 * 10 = 100

test = data[data_targets >= 30] 

test.shape
# Test on a subset of people

# Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10             # // is unconditionally "flooring division"

n_faces
face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids
# we have n_faces random-faces from within 1 to 100

test = test[face_ids, :]   

test.shape
# Total pixels in any image

n_pixels = data.shape[1]

n_pixels
# Select upper half of the faces as predictors

# // is unconditionally "flooring division"

#    3.1//1.2 = 2.0

X_train = train[:, :(n_pixels + 1) // 2]   

X_train
# Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]

y_train
# Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]
# Fit multi-output estimators

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "RandomForestRegressor": RandomForestRegressor()    

}
# Create an empty dictionary to collect prediction values

y_test_predict = dict()
# Fit each model by turn and make predictions

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)

    y_test_predict[name] = estimator.predict(X_test)
y_test_predict
y_test_predict['RandomForestRegressor'].shape
## Processing output

# Each face should have this dimension

image_shape = (64, 64)
# For 'Ridge' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))

j = 0

for i in range(n_faces):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))

    j = j+1

    plt.subplot(4,5,j)

    y = actual_face.reshape(image_shape)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,5,j)

    x = completed_face.reshape(image_shape)

    imshow(y)

  

plt.show()
# For 'Extra trees' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))

j = 0

for i in range(n_faces):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))

    j = j+1

    plt.subplot(4,5,j)

    y = actual_face.reshape(image_shape)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,5,j)

    x = completed_face.reshape(image_shape)

    imshow(y)

  

plt.show()
## For 'Linear regression' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))

j = 0

for i in range(n_faces):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))

    j = j+1

    plt.subplot(4,5,j)

    y = actual_face.reshape(image_shape)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,5,j)

    x = completed_face.reshape(image_shape)

    imshow(y)

  

plt.show()
# For '"K-nn' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))

j = 0

for i in range(5):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))

    j = j+1

    plt.subplot(4,5,j)

    y = actual_face.reshape(image_shape)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,5,j)

    x = completed_face.reshape(image_shape)

    imshow(y)

  

plt.show()
# For 'RandomForestRegressor' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))

j = 0

for i in range(5):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['RandomForestRegressor'][i]))

    j = j+1

    plt.subplot(4,5,j)

    y = actual_face.reshape(image_shape)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,5,j)

    x = completed_face.reshape(image_shape)

    imshow(y)

  

plt.show()