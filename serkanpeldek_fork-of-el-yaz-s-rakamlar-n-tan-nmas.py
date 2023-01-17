# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

#image processing tools
import cv2
import imutils
import mahotas

#Machine Learning tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
print(os.listdir("../input"))
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
#data=np.genfromtxt("../input/train.csv",delimiter=",", dtype="uint8")
#target=data[:,0]
#data=data[:,1:].reshape(data.shape[0], 28,28)
#############################
test_dataset=pd.read_csv("../input/test.csv",dtype="uint8")
train=pd.read_csv("../input/train.csv",dtype="uint8")
#train=np.genfromtxt("../input/train.csv", delimiter=",", dtype="uint8")

target=train['label']
data=train.drop('label',axis=1).values
test_dataset=test_dataset.values
#data=train[:,1:].reshape(train.shape[0], 28,28)

X_train=data[:,0:].reshape(data.shape[0], 28,28)
test_dataset=test_dataset[:,0:].reshape(test_dataset.shape[0],28,28)
#y_train=train[:,0]

print("train target shape:",target.shape)
print("train shape       :",X_train.shape)
print("test shape        :",test_dataset.shape)
def show_digit_matrix(digit, n=10):
    v_images=[]
    n=n
    count=0
    for i in range(0,n):
        h_images=list()
        for j in range(0,n):
            h_images.append(digit[count])
            count+=1
        h=np.hstack((h_images))
        v_images.append(h)
    image_matrix=np.vstack((v_images))
    
    fig, axarr = plt.subplots(1, 1, figsize=(12, 12))
    plt.imshow(image_matrix,cmap='gray')

show_digit_matrix(digit=X_train, n=20)
def deskew(image, width):
    (h,w)=image.shape[:2]
    
    #skew=mahotas.moments(img, p0, p1, cm=(0, 0), convert_to_float=True)
    
    moments=cv2.moments(image)
    skew=moments['mu11']/moments["mu02"]


    M=np.float32([
        [1, skew, -0.5*w*skew],
        [0,1,0]
    ])

    image_wrap=cv2.warpAffine(image, M, (w,h),
                         flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

    image_wrap=imutils.resize(image=image_wrap,width=width)

    return image_wrap

def center_extent(image, size):
    (eW, eH)=size

    if image.shape[1]>image.shape[0]:
        image=imutils.resize(image,width=eW)
    else:
        image=imutils.resize(image,height=eH)

    extent=np.zeros((eH,eW),dtype='uint8')

    offsetX=(eW-image.shape[1])//2
    offestY=(eH-image.shape[0])//2

    extent[offestY:offestY+image.shape[0],offsetX:offsetX+image.shape[1]]=image

    CM=mahotas.center_of_mass(extent)
    (cY, cX)=np.round(CM).astype("int32")
    (dX, dY)=((size[0]//2)-cX, (size[1]//2)-cY)
    M=np.float32([[1, 0, dX],
                  [0, 1, dY]])

    extent=cv2.warpAffine(extent, M, size)

    return extent
deskew_size=28
image=X_train[0]

image_deskewed=deskew(image, deskew_size)
image_extented=center_extent(image_deskewed,(deskew_size,deskew_size))
image_merged=np.hstack(([image,image_extented]))
plt.imshow(image_merged ,cmap='gray')
def process_data(mydata, use_hog=False):
    processed_data=[]


    deskew_size=20
    counter=0
    counter2=0
    print("ön işlem başladı:")
    for image in mydata:
        image_deskewed=deskew(image, deskew_size)
        image_extented=center_extent(image_deskewed,(deskew_size,deskew_size))

        processed_data.append(image_extented.flatten())
        
        if counter>10000:
            counter2+=1
            print(counter2*10000, end=", ")
            counter=0
            continue
        counter+=1
    print("ön işlem tamamlandı")
    return processed_data
    
processed_data=process_data(X_train)
X_train_processed, X_test_processed, y_train, y_test=train_test_split(processed_data, target, test_size=0.3, random_state=42)

#Aşağıda işlemler uzun sürmektedir. elde edilen en iyi sonuç
#n=5 için
# test accuracy : 0.9788888888888889
"""
neigbors=range(1,15)
test_accuracy=list()
train_accuracy=[]
counter=0
for n in neigbors:
    clf=KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train_processed, y_train)
    #train_accuracy.append(clf.score(X_train_processed, y_train))
    test_accuracy.append(clf.score(X_test_processed, y_test))
    print("for n :",n)
    #print("train accuracy:",train_accuracy[counter])
    print("test accuracy :",test_accuracy[counter])
    print()
    counter+=1
"""
#sonuçlar
"""
sonuçlar
for n : 1
test accuracy : 0.9773809523809524

for n : 2
test accuracy : 0.976031746031746

for n : 3
test accuracy : 0.9788888888888889

for n : 4
test accuracy : 0.9783333333333334

for n : 5
test accuracy : 0.9788888888888889

for n : 6
test accuracy : 0.9784920634920635

for n : 7
test accuracy : 0.9766666666666667

for n : 8
test accuracy : 0.9768253968253968

for n : 9
test accuracy : 0.9753968253968254

for n : 10
test accuracy : 0.976031746031746

for n : 11
test accuracy : 0.9739682539682539

for n : 12
test accuracy : 0.9744444444444444

for n : 13
test accuracy : 0.972936507936508

for n : 14
test accuracy : 0.973015873015873
"""
n=5
clf=KNeighborsClassifier(n_neighbors=n)
clf.fit(X_train_processed, y_train)
y_pred1=clf.predict(X_test_processed)
print("for {} accuracy:{}".format(n, accuracy_score(y_pred1, y_test)))
print("for {} confusion matrix:\n{}".format(n, confusion_matrix(y_pred1, y_test)))
print("for {} classification reports:\n{}".format(n, classification_report(y_pred1, y_test)))
processed_test_data=process_data(test_dataset)
print("gönderi hazırlanıyor...")
clf=KNeighborsClassifier(n_neighbors=5)
clf.fit(processed_data, target)
y_pred2=clf.predict(processed_test_data)
print("gönderi hazır.")
results = pd.Series(y_pred2,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("preprocess_and_knn(n=5)_mnist2.csv",index=False)
print("gönderi kaydedildi2.")