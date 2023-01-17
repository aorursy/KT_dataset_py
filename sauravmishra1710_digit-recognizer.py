# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Install the additional packages required to run the notebook



!pip install mahotas

!pip install pypng
# Import all the necessary library and pacgakes



import os,png,array

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn import svm

from sklearn.svm import LinearSVC

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



import gc

import warnings



from PIL import Image

from skimage import feature



import cv2

import mahotas

import _pickle as cPickle
# Initialize notebook parameters



# To display all the columns

pd.options.display.max_columns = None



# To display all the rows

pd.options.display.max_rows = None



# To map Empty Strings or numpy.inf as Na Values

pd.options.mode.use_inf_as_na = True



# Set Precision to 8 for better readability

pd.set_option('precision', 8)

pd.options.display.float_format = '{:.4f}'.format



pd.options.display.expand_frame_repr =  False



%matplotlib inline



# Set Style

sns.set(style = "whitegrid")



# Ignore Warnings

warnings.filterwarnings('ignore')
# Variable declaration

HOG_feature_data = []

test_data = False
# import time



# os.chdir('C:/Hackathons/Digits_Identification-AnalyticsVidya/Train_UQcUa52/Images/train')



# Creating a list of columns to represent the pixel format for a particular image.



# columnNames = list()

# for i in range(784):

#     pixel = 'pixel'

#     pixel += str(i)

#     columnNames.append(pixel)



# pixel_data = pd.DataFrame(columns = columnNames)

# start_time = time.time()

# for i in range(1,49000):

#     img_name = str(i)+'.png'

#     img = Image.open(img_name)

#     rawData = img.load()

#     data = []

#     for y in range(28):

#         for x in range(28):

#             data.append(rawData[x,y][0])

#     k = 0

#     pixel_data.loc[i] = [data[k] for k in range(784)]



# The below 6 lines of code does not need to be executed for test image set as we do not want the label column.

# **************************************************************************************************************

# label_data = pd.read_csv("train.csv")

# train_labels = label_data['label']

# pixel_data = pd.concat([label_data, pixel_data],axis = 1)

# pixel_data = pixel_data.drop('filename',1)

# pixel_data = pixel_data.dropna()

# **************************************************************************************************************



# Convert the pixel dataset into a csv file.

# pixel_data.to_csv("image_pixel_data.csv",index = False)



# Change the directory to the original directory. This could change depending on the directory level.

# os.chdir('../../../')

# print("Conversion completed in  - ", time.time()-start_time)
def getdigits(dfPath):

    

    if test_data == True:

        # build the dataset. We do not have the label column in case of test data. 

        # Hence we would not get the target/labels and return 0 for the same.

        data = np.genfromtxt(dfPath, delimiter = ",", dtype = "uint8")

        data = data[:, 0:].reshape(data.shape[0], 28, 28)

        # return a tuple of the data and targets

        return (data, 0)

    else:

        # build the dataset and then split it into data and labels (target)

        data = np.genfromtxt(dfPath, delimiter = ",", dtype = "uint8")

        target = data[:, 0]

        data = data[:, 1:].reshape(data.shape[0], 28, 28)

        # return a tuple of the data and targets

        return (data, target)
test_data = False

(imgdata,label) = getdigits('/kaggle/input/digit-recognizer/train.csv')



%matplotlib inline





for i in range(1, 21):

  plt.subplot(4,5,i)

  plt.tight_layout()

  plt.imshow(imgdata[i], cmap='gray', interpolation='none')

  plt.title("Digit: {}".format(label[i]))

  plt.xticks([])

  plt.yticks([])

plt.show()
digits_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

digits_df.head()
digit_one = digits_df.iloc[0, 1:]

digit_one.shape
digit_one = digit_one.values.reshape(28, 28)

plt.imshow(digit_one, cmap='gray')
print(digit_one[5:-5, 5:-5])
def deskewImage(image, width):

    # grab the width and height of the image and compute

    # moments for the image

    (h, w) = image.shape[:2]

    moments = cv2.moments(image)

    

    # deskew the image by applying an affine transformation

    skew = moments["mu11"] / moments["mu02"]

    M = np.float32([

        [1, skew, -0.5 * w * skew],

        [0, 1, 0]])

    image = cv2.warpAffine(image, M, (w, h),

        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)



    # resize the image to have a constant width

    image = cv2.resize(image, (28,28))

    

    # return the deskewed image

    return image
deSkewedImage = deskewImage(imgdata[8],28)



plt.subplot(1,2,1)

plt.tight_layout()

plt.imshow(imgdata[8], cmap='gray', interpolation='none')

plt.title("Skewed Image")



plt.subplot(1,2,2)

plt.tight_layout()

plt.imshow(deSkewedImage, cmap='gray', interpolation='none')

plt.title("De-Skewed Image")



plt.show()
def centre_align_image(image, size):

    # grab the extent width and height

    (eW, eH) = size

    

    #Image Shape is

    (h, w) = image.shape[:2]

    

    #New dimension according to image aspect ratio

    dim = None

    

    # handle when the width is greater than the height

    if image.shape[1] > image.shape[0]:

        #image = resize(image, width = eW)

        r = eW / float(w)

        dim = (eW, int(h * r))

        image = cv2.resize(image,dim,cv2.INTER_AREA)



    # otherwise, the height is greater than the width

    else:

        #image = resize(image, height = eH)

        r = eH / float(h)

        dim = (int(w * r), eH)

        image = cv2.resize(image,dim,cv2.INTER_AREA)



    # allocate memory for the extent of the image and grab it

    extent = np.zeros((eH, eW), dtype = "uint8")

    offsetX = (eW - image.shape[1]) / 2

    offsetY = (eH - image.shape[0]) / 2

    offsetX = int(offsetX)

    offsetY = int(offsetY)

    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image



    # compute the center of mass of the image and then

    # move the center of mass to the center of the image

    (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")

    (dX, dY) = ((size[0] / 2) - cX, (size[1] / 2) - cY)

    M = np.float32([[1, 0, dX], [0, 1, dY]])

    extent = cv2.warpAffine(extent, M, size)



    # return the extent of the image

    return extent
# An example transformation on a de-centered image.



img = cv2.imread('/kaggle/input/digit-recognizer-images/Decenter.png')

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

aligned_image = centre_align_image(img,(28,28))



plt.subplot(1,2,1)

plt.tight_layout()

plt.imshow(img, cmap='gray', interpolation='none')

plt.title("De-Centered Image")



plt.subplot(1,2,2)

plt.tight_layout()

plt.imshow(aligned_image, cmap='gray', interpolation='none')

plt.title("Centered Image")



plt.show()
def extract_HOG_features(image):

    hist = feature.hog(image, orientations = 9,

        pixels_per_cell = (8, 8),

        cells_per_block = (3, 3)

        )

    return hist
hist = extract_HOG_features(imgdata[0])



print(np.shape(hist))
# This method transforms an image to a proper format so as to increase the performance of the classifier.

def processImages(filePath):

    image_size = 28

    

    (digits, target) = getdigits(filePath)



    # loop over the images

    for image in digits:

        # deskew the image, center it

        image = deskewImage(image, image_size)

        image = centre_align_image(image, (image_size, image_size))

    

        # describe the image and update the data matrix

        hist = extract_HOG_features(image)

        HOG_feature_data.append(hist)
#call the above function tp apply the transformations.

test_data = False

processImages('/kaggle/input/digit-recognizer/train.csv')
print("Total images processed - ", len(HOG_feature_data))

print("HOG features extracted per image - ", len(HOG_feature_data[0]))
# Let's check the total number of observations present for each digit.

# Our label field represents numbers (int) from 1 to 9. But the label column is actually a categorical

# column here. Let's convert 'label' field as categorical for a smoother and correct analysis.



digits_df.label = digits_df.label.astype('category')



digits_df.label.value_counts()
# Summarise count in terms of percentage 

(round(digits_df.label.value_counts()/len(digits_df.index), 4))*100
# average values/distributions of features

description = digits_df.describe()

description
# Creating training and test sets

# Splitting the data into train and test

X = digits_df.iloc[:, 1:]

y = digits_df.iloc[:, 0]



# Rescaling the features

X = scale(X)



# train test split with train_size=10% and test size=90%

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=101)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

linear_SVM = svm.SVC(kernel='linear')



# fit

linear_SVM.fit(X_train, y_train)
# predict

predictions = linear_SVM.predict(X_test)

predictions[:10]
confusion_matrix = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

confusion_matrix
metrics.accuracy_score(y_true=y_test, y_pred=predictions)
# class-wise accuracy

class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)

print(class_wise)
# run gc.collect() (garbage collect) to free up memory

# else, since the dataset is large and SVM is computationally heavy,

# it'll throw a memory error while training

gc.collect()
# rbf kernel with other hyperparameters kept to default 

svm_rbf = svm.SVC(kernel='rbf')

svm_rbf.fit(X_train, y_train)
# predict

predictions = svm_rbf.predict(X_test)

predictions[:10]
# accuracy 

print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))
# conduct (grid search) cross-validation to find the optimal values 

# of cost C and the choice of kernel



# parameters = {'C':[1, 10, 100], 

#              'gamma': [1e-2, 1e-3, 1e-4]}



# # instantiate a model 

# svc_grid_search = svm.SVC(kernel="rbf")



# # create a classifier to perform grid search

# gs_cv = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy')



# # fit

# gs_cv.fit(X_train, y_train)
# results

# gs_cv_results = pd.DataFrame(gs_cv.cv_results_)

# gs_cv_results
# converting C to numeric type for plotting on x-axis

# gs_cv_results['param_C'] = gs_cv_results['param_C'].astype('int')



# # # plotting

# plt.figure(figsize=(16,6))



# # subplot 1/3

# plt.subplot(1,3,1)

# gamma_01 = gs_cv_results[gs_cv_results['param_gamma'] == 0.01]



# plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

# plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.01")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')



# # subplot 2/3

# plt.subplot(1,3,2)

# gamma_001 = gs_cv_results[gs_cv_results['param_gamma'] == 0.001]



# plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

# plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.001")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')





# # subplot 3/3

# plt.subplot(1,3,3)

# gamma_0001 = gs_cv_results[gs_cv_results['param_gamma'] == 0.0001]



# plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

# plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

# plt.xlabel('C')

# plt.ylabel('Accuracy')

# plt.title("Gamma=0.0001")

# plt.ylim([0.60, 1])

# plt.legend(['test accuracy', 'train accuracy'], loc='lower right')

# plt.xscale('log')



# plt.show()
svm_poly = svm.SVC(kernel='poly', C=100, gamma='auto', degree=3, coef0=1, decision_function_shape='ovo')



svm_poly.fit(X_train, y_train)
poly_predictions = svm_poly.predict(X_test)

poly_predictions[:10]
# evaluation: CM 

poly_confusion = metrics.confusion_matrix(y_true = y_test, y_pred = poly_predictions)



# measure accuracy

poly_test_accuracy = metrics.accuracy_score(y_true = y_test, y_pred = poly_predictions)



print("Accuracy - ", poly_test_accuracy, "\n")

print(poly_confusion)
test_data = True

processImages('/kaggle/input/digit-recognizer/test.csv')
processed_test_set_df_kaggle = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



# Rescaling the features

processed_test_set_df_kaggle = scale(processed_test_set_df_kaggle)
# final predict Kaggle

final_predictions_kaggle = svm_poly.predict(processed_test_set_df_kaggle)

final_predictions_kaggle[:10]
image_ids = []

for i in range(1, 28001):

    image_ids.append(i)



submission_file_df = pd.DataFrame({'ImageId': image_ids, 'Label': final_predictions_kaggle})



filename = 'submission_Kaggle.csv'



submission_file_df.to_csv(filename, index = False, sep = ",")