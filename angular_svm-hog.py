import numpy as np
import cv2
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
# it works fine on pc but causing problem here to fetch this file
mnist = fetch_openml("mnist_784")

X, y = mnist['data'], mnist['target']
data = np.array(X, 'int16')
target = np.array(y, 'int')
list_hog = []
for feature in data:
 fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(14,14),cells_per_block=(1,1),visualize=False )
 list_hog.append(fd)
hog_features = np.array(list_hog, 'float64')
preProcess = preprocessing.MaxAbsScaler().fit(hog_features)
hog_features_transformed = preProcess.transform(hog_features)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hog_features_transformed,target , random_state = 0)
Model= SVC(C=3.0, kernel='rbf')
print("Training Started")
Model.fit(X_train, y_train)
print("Training Complete")
print(Model.score(X_test, y_test))

#Following code can be used to predict a digit after showing image that isnot part o

#joblib.dump((Model, preProcess), "ModelDigit42.pkl", compress=3)
#print("prediction has started")
#Model, preProcess = joblib.load("ModelDigit42.pkl")
#img = cv2.imread("test.jpg")
# Convert to grayscale and apply Gaussian filtering
#im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#im_gray = cv2.GaussianBlur(im_grey, (5, 5), 0)
#ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
# Find contours in the image
#ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
#rects = [cv2.boundingRect(ctr) for ctr in ctrs]

clear
