import gc

gc.collect()

# Libraries

import numpy as np

import matplotlib.pyplot as plt

from skimage.io import imshow

from sklearn.datasets import fetch_olivetti_faces

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

data = np.load("../input/olivetti_faces.npy")

type(data)

targets = np.load("../input/olivetti_faces_target.npy")

type(targets)

targets.size

data.shape

firstImage = data[0]

imshow(firstImage)

# Flatten each image

data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

data.shape

# Patition datasets into two (fancy indexing)

targets < 30                # Output is true/false

train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100



n_faces = test.shape[0]//10             # // is unconditionally "flooring division",

n_faces

face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids

# 7.1 So we have n_faces random-faces from within 1 to 100

test = test[face_ids, :]



# Total pixels in any image

n_pixels = data.shape[1]



# Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",

                                            #    3.1//1.2 = 2.0

# Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]



# Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]



# Fit multi-output estimators

# Prepare a dictionary of estimators after instantiating each one of them

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

}





# Create an empty dictionary to collect prediction values

y_test_predict = dict()



# Fit each model by turn and make predictions

#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    

    y_test_predict[name] = estimator.predict(X_test)   

                                                         

y_test_predict



y_test_predict['Ridge'].shape



image_shape = (64, 64)



#  Total faces per estimator: 2 * n_faces



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