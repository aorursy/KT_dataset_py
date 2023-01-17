# 1.0 Call libraries

# For data manipulation

import numpy as np



# 1.1 For plotting faces

import matplotlib.pyplot as plt   

from skimage.io import imshow



# 1.2 For dataset

from sklearn.datasets import fetch_olivetti_faces



# 1.3 Regressors

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor

image_data = np.load("../input/olivetti_faces.npy")

image_data.shape
targets = np.load("../input/olivetti_faces_target.npy")

targets.size 
image_data                    # Images set

image_data.shape              # Image is 400X 64 X 64



# 4.1 See an image

firstImage = image_data[0]

imshow(firstImage) 
data = image_data.reshape(image_data.shape[0], image_data.shape[1] * image_data.shape[2])     # 64 X 64 = 4096

# 5.1 Flattened 64 X 64 array

data.shape                                # 400 X 4096
targets < 30                # Output is true/false

train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100

test.shape
# Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10            

print("no of Faces =",n_faces)

face_ids = np.random.randint(0 , 100, size =n_faces)

print("Random Face Ids =",face_ids)

# So we have n_faces random-faces from within 1 to 100

test = test[face_ids, :]  
# Total pixels in any image

n_pixels = data.shape[1]



# Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",

                                            #    3.1//1.2 = 2.0

# Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]



#Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]
# Prepare a dictionary of estimators after instantiating each one of them

# Prepare a dictionary of estimators after instantiating each one of them

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,     # Out of 20000

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),                          # Accept default parameters

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "RandomForestRegressor": RandomForestRegressor()  

}



# Create an empty dictionary to collect prediction values

y_test_predict = dict()
#Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

                                                       # Note that output of estimator.predict(X_test) is prediction for

                                                       #  all the test images and NOT one (or one-by-one)

# Just verify    

y_test_predict



# Just check shape of one of them

y_test_predict['Ridge'].shape    # 5 X 2048 
# Each face should have this dimension

image_shape = (64, 64)

## For 'Ridge' regression

##      We will have total images as follows:

#      Per esimator, we will have n_faces * 2

#      So total - n_estimators * n_faces * 2

#      Fig size should be accordingly drawn



# Total faces per estimator: 2 * n_faces



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
plt.figure(figsize=(  n_faces * 1, n_faces))

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