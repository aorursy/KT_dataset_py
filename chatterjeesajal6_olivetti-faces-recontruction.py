## Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For plotting faces

import matplotlib.pyplot as plt   

from skimage.io import imshow

# The dataset is here

from sklearn.datasets import fetch_olivetti_faces



# Regressors

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

# Read the Input data files consisting of the images

img_data = np.load("../input/olivetti_faces.npy")

type(img_data)

img_data.shape
# Extract data components

# Target :: There are 400 targets

targets = np.load("../input/olivetti_faces_target.npy")

targets.size
# Plot 10 images of same person from the data 

def draw_faces(idx,ncol,nrow,pos):

    for i in range(nrow):

        pos +=1

        ax = plt.subplot(ncol,nrow,pos)

        ax.set_title('Image '+str(ncol))

        imshow(img_data[pos-1])

    return(pos)

    

fig,ax = plt.subplots(1,1,figsize=(20,20))

j=0

pos=0

for i in range(1):

    j +=1

    npos = draw_faces(i,j,10,pos)

    pos = npos + pos
# Flatten each image

data = img_data.reshape(img_data.shape[0], img_data.shape[1] * img_data.shape[2])     # 64 X 64 = 4096

# Flattened 64 X 64 array

data.shape                                # 400 X 4096
# Patition datasets into two (fancy indexing)

targets < 30                # Output is true/false

train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100



# Test on a subset of people

#     Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10             # // is unconditionally "flooring division",

n_faces

face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids

# So we have n_faces random-faces from within 1 to 100

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

#  Experiment with more mulit-output regressors (such as RandomForestRegressor)

# Prepare a dictionary of estimators after instantiating each one of them

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,     # Out of 20000

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),                          # Accept default parameters

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

}

# Create an empty dictionary to collect prediction values

y_test_predict = dict()



# Fit each model by turn and make predictions

#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

                                                       # Note that output of estimator.predict(X_test) is prediction for

                                                       #  all the test images and NOT one (or one-by-one)

# Display the predicted data    

y_test_predict

## Processing output

# Each face should have this dimension

image_shape = (64, 64)



# number of faces

n_faces
def plot_Reg_images(Regre_name):

    j = 0

    for i in range(n_faces):

        actual_face =    test[i].reshape(image_shape)

        completed_face = np.hstack((X_test[i], y_test_predict[Regre_name][i]))

        j = j+1

        plt.subplot(4,10,j)

        y = actual_face.reshape(image_shape)

        imshow(y)

        j = j+1

        plt.subplot(4,10,j)

        x = completed_face.reshape(image_shape)

        imshow(x)

    plt.show()
##  For 'Ridge' regression

##      We will pass 'Ridge' for plotting images contructed through Ridge regression 



plt.figure(figsize=( 20, 20))

Regression_method = 'Ridge'

plot_Reg_images(Regression_method)



##  For 'Extra trees' regression

##      We will pass 'Extra trees' for plotting images contructed through Ridge regression 



plt.figure(figsize=( 20, 20))

Regression_method = 'Extra trees'

plot_Reg_images(Regression_method)

##  For 'Linear regression' regression

##      We will pass 'Linear regression' for plotting images contructed through Ridge regression 



plt.figure(figsize=( 20, 20))

Regression_method = 'Linear regression'

plot_Reg_images(Regression_method)

##  For For 'K-nn' regression regression

##      We will pass 'K-nn' for plotting images contructed through Ridge regression 



plt.figure(figsize=( 20, 20))

Regression_method = 'K-nn'

plot_Reg_images(Regression_method)
