#Load Libraries

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
img_data = np.load("../input/olivetti_faces.npy")

img_data.shape 
'''There are ten different images of each of 40 distinct subjects.

That is each subject has 10 images but taken differently

For some subjects, the images were taken at different times, varying the lighting, facial expressions

(open / closed eyes,smiling / not smiling) and facial details (glasses / no glasses).

All the images were taken against a dark homogeneous background

with the subjects in an upright, frontal position (with tolerance  for some side     movement) '''



def printSubject(subject):

    plt.figure(figsize=(20,20))

    for i in range (10):

        plt.subplot(1, 10, i+1)

        x = img_data[i+subject]

        imshow(x)

    plt.show()



#plot first and last subject

subject1 = 0

subject40 = 390



printSubject(subject1)

printSubject(subject40)
target = np.load("../input/olivetti_faces_target.npy")

target.shape
  # 64 X 64 = 4096

img_data_flat = img_data.reshape(img_data.shape[0], img_data.shape[1] * img_data.shape[2])   

img_data_flat.shape
#Patition datasets into two (fancy indexing)

target < 30                # Output is true/false

train = img_data_flat[target < 30]  # First 30 types of images out of 40 ie 30 * 10 =300

test = img_data_flat[target >= 30]  # Test on rest independent people  10 * 10 = 100
#Generate 10 random integers between 0 and 100

n_faces = test.shape[0]//10             # // is unconditionally "flooring division (integer division)",

n_faces

face_ids = np.random.randint(0 , 100, size =n_faces)

#So we have n_faces which are random from within 1 to 100

test = test[face_ids, :]  

face_ids
#Total pixels in any image

n_pixels = img_data_flat.shape[1]



#Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",

                                            #    3.1//1.2 = 2.0

#Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]



#Similarly for test data. Upper and lower half

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

}



#Create an empty dictionary to collect prediction values

y_test_predict = dict()



#Fit each model by turn and make predictions

#     Iterate over dict items. Each item is a tuple: ( name,estimator-object)s

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)                    # fit() with instantiated object

    y_test_predict[name] = estimator.predict(X_test)   # Make predictions and save it in dict under key: name

                                                       # Note that output of estimator.predict(X_test) is prediction for

                                                       #  all the test images and NOT one (or one-by-one)

#Just check    

y_test_predict

#Each face should have this dimension

image_shape = (64, 64)



def regressionPlot(Regesion):

    plt.figure(figsize=(10, 10))

    j = 0

    for i in range(n_faces):

        actual_face =    test[i].reshape(image_shape)

        completed_face = np.hstack((X_test[i], y_test_predict[Regesion][i]))

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
## For 'Ridge' regression

## We will have total images as follows:

#  Per esimator, we will have n_faces * 2

#  So total - n_estimators * n_faces * 2

#  Fig size should be accordingly drawn



# Total faces per estimator: 2 * n_faces

regressionPlot('Ridge')
regressionPlot('Extra trees')
regressionPlot('Linear regression')
regressionPlot('K-nn')