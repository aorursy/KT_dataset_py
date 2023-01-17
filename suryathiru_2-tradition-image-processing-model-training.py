import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import os
import pydicom as dcm
np.random.seed(42)
data = pd.read_csv('../input/1-tradition-image-processing-feature-extraction/img_features.csv')
data.head()
feats = data.features.apply(lambda x: list(eval(x)))  # oops!
dataset = pd.DataFrame(feats.values.tolist(), 
                        columns=['mean', 'stddev', 'area', 'perimeter', 'irregularity', 'equiv_diam', 'hu1', 'hu2', 'hu4', 'hu5', 'hu6'],
                       index=data.index)
dataset['label'] = data['target']
dataset.head()
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('label', axis=1), dataset['label'],
                                                   test_size=0.4, stratify=dataset['label'])
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def print_metrics(y_pred, y_train, yt_pred, y_test):
    print('Train data metrics:')
    print('Accuracy: ', accuracy_score(y_train, y_pred))
    print('Precison: ', precision_score(y_train, y_pred))
    print('Recall: ', recall_score(y_train, y_pred))
    print('F1 score: ', f1_score(y_train, y_pred))
    print('ROC AUC score: ', roc_auc_score(y_train, y_pred))
    print()
    print('Test data metrics:')
    print('Accuracy: ', accuracy_score(y_test, yt_pred))
    print('Precison: ', precision_score(y_test, yt_pred))
    print('Recall: ', recall_score(y_test, yt_pred))
    print('F1 score: ', f1_score(y_test, yt_pred))
    print('ROC AUC score: ', roc_auc_score(y_test, yt_pred))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

print_metrics(lr.predict(X_train), y_train, lr.predict(X_test), y_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(500, max_depth=8, min_samples_split=2,
                            n_jobs=-1)
rf.fit(X_train, y_train)

print_metrics(rf.predict(X_train), y_train, rf.predict(X_test), y_test)
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                               max_depth=7, min_samples_split=5)
gb.fit(X_train, y_train)

print_metrics(gb.predict(X_train), y_train, gb.predict(X_test), y_test)
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

print_metrics(svm.predict(X_train), y_train, svm.predict(X_test), y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(20)
knn.fit(X_train, y_train)

print_metrics(knn.predict(X_train), y_train, knn.predict(X_test), y_test)
from yellowbrick.classifier import ConfusionMatrix

cm = ConfusionMatrix(gb, classes=[0,1], 
                     encoder={0: 'normal', 1: 'pneumonia'})
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(gb, classes=["normal", "pneumonia"])

visualizer.fit(X_train, y_train) 
visualizer.score(X_test, y_test) 
visualizer.show()    
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure(figsize=(13,7))
plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlim([-1, X_train.shape[1]])
import skimage
from skimage import feature, filters
PATH = '../input/rsna-pneumonia-detection-challenge'

def load_image(path):    
    patientImage = path + '.dcm'
    imagePath = os.path.join(PATH,"stage_2_train_images/", patientImage)
    img = dcm.read_file(imagePath).pixel_array
    return img

def imshow_gray(img):
    plt.figure(figsize=(12,7))
    return plt.imshow(img, cmap='gray')

def area(img):
    # binarized image as input
    return np.count_nonzero(img)

def perimeter(img):
    # edges of the image as input
    return np.count_nonzero(img)

def irregularity(area, perimeter):
    # area and perimeter of the image as input, also called compactness
    I = (4 * np.pi * area) / (perimeter ** 2)
    return I

def equiv_diam(area):
    # area of image as input
    ed = np.sqrt((4 * area) / np.pi)
    return ed

def get_hu_moments(contour):
    # hu moments except 3rd and 7th (5 values)
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).ravel().tolist()
    del hu[2]
    del hu[-1]
    log_hu = [-np.sign(a)*np.log10(np.abs(a)) for a in hu]
    return log_hu


def extract_features(img):
    mean = img.mean()
    std_dev = img.std()
    
    # hist equalization
    equalized = cv2.equalizeHist(img)
    
    # sharpening
    hpf_kernel = np.full((3, 3), -1)
    hpf_kernel[1,1] = 9
    sharpened = cv2.filter2D(equalized, -1, hpf_kernel)
    
    # thresholding
    ret, binarized = cv2.threshold(cv2.GaussianBlur(sharpened,(7,7),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # edge detection
    edges = skimage.filters.sobel(binarized)
    
    # moments from contours
    contours, hier = cv2.findContours((edges * 255).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    select_contour = sorted(contours, key=lambda x: x.shape[0], reverse=True)[0]
    
    
    # feature extraction
    ar = area(binarized)
    per = perimeter(edges)
    irreg = irregularity(ar, per)
    eq_diam = equiv_diam(ar)
    hu = get_hu_moments(select_contour)
    
    return (mean, std_dev, ar, per, irreg, eq_diam, *hu)
test_img = data[data['target']==1]['patientId'].sample(1)

img = load_image(test_img.values[0])
imshow_gray(img)
feats = list(extract_features(img))

pred = gb.predict([feats])

if pred == 1:
    print('Patient is infected with pneumonia')
else:
    print('Patient is normal')
from skimage.util import random_noise

img = random_noise(img, mode='gaussian')
img = (img*255).astype('int')
imshow_gray(img)
feats = list(extract_features(img.astype('uint8')))

pred = gb.predict([feats])

if pred == 1:
    print('Patient is infected with pneumonia')
else:
    print('Patient is normal')