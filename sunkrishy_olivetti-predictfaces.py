# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt   

from skimage.io import imshow


from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor
images = np.load("../input/olivetti_faces.npy")

images.shape



# Extract data components

targets = np.load("../input/olivetti_faces_target.npy")   # Data target

type(targets)

targets.size                   # 400 images
#  See an image

firstImage = images[0]

imshow(firstImage) 


data = images.reshape(images.shape[0], images.shape[1] * images.shape[2])     # 64 X 64 = 4096



# Flattened 64 X 64 array

data.shape                                # 400 X 4096

#  Patition datasets into two (fancy indexing)

targets < 30                # Output is true/false



# First 30 types of images out of 40 ie 30 * 10 =300

train = data[targets < 30]  

# Test on rest independent people  10 * 10 = 100

test = data[targets >= 30]  



#     Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10             # // is unconditionally "flooring division",

n_faces



face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids



# So we have n_faces random-faces from within 1 to 100

test = test[face_ids, :]   



#  Total pixels in any image

n_pixels = data.shape[1]

n_pixels
X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",

                                            #    3.1//1.2 = 2.0

X_train



# Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]

y_train



# Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]
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



#  Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

import os, time, sys

start = time.time()

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

                                                       # Note that output of estimator.predict(X_test) is prediction for

                                                       #  all the test images and NOT one (or one-by-one)

end = time.time()

(end-start)/60
# Few checks    

y_test_predict

y_test_predict['Ridge'].shape    # 5 X 2048    

y_test_predict['RandomForestRegressor'].shape



## Processing output Each face should have this dimension

image_shape = (64, 64)
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
# For 'RandomForestRegressor' regressionplt.figure(figsize=( 2 * n_faces * 2, 5))

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