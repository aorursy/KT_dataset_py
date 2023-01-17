### Call Libraries 

import numpy as np

import matplotlib.pyplot as plt   

from skimage.io import imshow

from sklearn.datasets import fetch_olivetti_faces

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor



### 

data = np.load("../input/olivetti_faces.npy")

data.shape



# 3 Extract data components

targets = np.load("../input/olivetti_faces_target.npy")

targets.shape        



# 4. Images next

data                    # Images set

data.shape              # Image is 400X 64 X 64



# 4.1 See an image

firstImage = data[0]

imshow(firstImage) 





# 5.0 Flatten each image

data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])     # 64 X 64 = 4096

# 5.1 Flattened 64 X 64 array

data.shape                                # 400 X 4096



# 6.0 Patition datasets into two (fancy indexing)

targets < 30                # Output is true/false

train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100





# 7.0 Test on a subset of people

#     Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10             # // is unconditionally "flooring division",

n_faces

face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids

# 7.1 So we have n_faces random-faces from within 1 to 100

test = test[face_ids, :]   





# 8.0 Total pixels in any image

n_pixels = data.shape[1]



# 8.1 Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",

                                            #    3.1//1.2 = 2.0

# 8.2 Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]



# 9.0 Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]



# 9. Fit multi-output estimators

#  Experiment with more mulit-output regressors (such as RandomForestRegressor)

#  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble



# Prepare a dictionary of estimators after instantiating each one of them

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,     # Out of 20000

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),                          # Accept default parameters

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "Lasso": Lasso(),

    "ElasticNet": ElasticNet(random_state=0),

    "RandomForestRegressor": RandomForestRegressor()    

}





# 9.1 Create an empty dictionary to collect prediction values

y_test_predict = dict()



# 10. Fit each model by turn and make predictions

#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name



# 10.1 Just check    

y_test_predict



# 10.2 Just check shape of one of them

y_test_predict['Ridge'].shape    # 5 X 2048    



## Processing output

# 11. Each face should have this dimension

image_shape = (64, 64)



## 11.1 For 'Ridge' regression

##      We will have total images as follows:

#      Per esimator, we will have n_faces * 2

#      So total - n_estimators * n_faces * 2

#      Fig size should be accordingly drawn



# 11.2 Total faces per estimator: 2 * n_faces

for name, regression in ESTIMATORS.items() :

    fig = plt.figure(figsize=( 2 * n_faces * 2, 5))

    fig.suptitle(name, fontsize=10)

    j = 0

    for i in range(n_faces):

        actual_face =    test[i].reshape(image_shape)

        completed_face = np.hstack((X_test[i], y_test_predict[name][i]))

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

    


