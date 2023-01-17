# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")
df_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test.csv")
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data,exposure
import cv2
from sklearn import svm
%matplotlib inline
np.random.seed(1)
train_x = df_train[df_train.columns[1::]].to_numpy()           
train_y = df_train[df_train.columns[0]].to_numpy()

test_x = df_test[df_test.columns[1::]].to_numpy() 
test_y = df_test[df_test.columns[0]].to_numpy()               

print("SUMMARY OF DATA:")

print("train_x shape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x shape: " + str(test_x.shape))
print("test_y shape: " + str(test_y.shape))
train_x = train_x/255
test_x = test_x/255
index = 4
plt.imshow(train_x[index].reshape(28, 28),cmap='gray')
plt.show()
from skimage.feature import hog
from skimage import data,exposure
import cv2
def get_hog(image):
    fd,hog_image=hog(image.reshape(28,28),pixels_per_cell=(2,2),
                        cells_per_block=(1, 1),visualize=True,feature_vector=False)
    return(fd,hog_image)

def show_hog(image,hog_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image.reshape(28, 28), cmap=plt.cm.gray)
    ax1.set_title('Input image')
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
df_htrain = pd.read_csv("../input/sl-hog-features/Hog_Features.csv")
df_htest = pd.read_csv("../input/sl-hog-features/Hog_Features_test.csv")
htrain = df_htrain[df_htrain.columns[1::]].to_numpy()
htest = df_htest[df_htest.columns[1::]].to_numpy()
print("Training Data size : ",htrain.shape)
print("Test Data size : ",htest.shape)
a,b=get_hog(train_x[17])
show_hog(train_x[17],b)
a=a.reshape(-1)
a=a.reshape(1,len(a))
print(a.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=30)
htrain_pca = pca.fit_transform(htrain)
htest_pca=pca.transform(htest)
print(htest_pca.shape)
htrain_pca.shape
clf = svm.SVC(kernel='linear') # rbf Kernel

#Train the model using the training sets
clf.fit(htrain_pca,train_y)

#Predict the response for test dataset
y_pred = clf.predict(htest_pca)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("F1 Score:",metrics.f1_score(test_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(test_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(test_y,y_pred,average = 'weighted'))
clf = svm.SVC(kernel='poly') # rbf Kernel

#Train the model using the training sets
clf.fit(htrain_pca,train_y)

#Predict the response for test dataset
y_pred = clf.predict(htest_pca)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("F1 Score:",metrics.f1_score(test_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(test_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(test_y,y_pred,average = 'weighted'))
clf = svm.SVC(kernel='sigmoid') # rbf Kernel

#Train the model using the training sets
clf.fit(htrain_pca,train_y)

#Predict the response for test dataset
y_pred = clf.predict(htest_pca)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("F1 Score:",metrics.f1_score(test_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(test_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(test_y,y_pred,average = 'weighted'))
clf = svm.SVC(kernel='rbf') # rbf Kernel

#Train the model using the training sets
clf.fit(htrain_pca,train_y)

#Predict the response for test dataset
y_pred = clf.predict(htest_pca)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("F1 Score:",metrics.f1_score(test_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(test_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(test_y,y_pred,average = 'weighted'))
y_pred = clf.predict(htrain_pca)
print("Training Set Metrics : ")
print("Accuracy:",metrics.accuracy_score(train_y, y_pred))
print("F1 Score:",metrics.f1_score(train_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(train_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(train_y,y_pred,average = 'weighted'))
x=[]
y=[]
y2=[]
for i in range(1,10):
    clf = svm.SVC(kernel='rbf',C=i)
    clf.fit(htrain_pca,train_y)
    y_pred = clf.predict(htest_pca)
    x.append(i)
    y.append(metrics.f1_score(test_y, y_pred, average='weighted'))
    y2_pred = clf.predict(htrain_pca)
    y2.append(metrics.f1_score(train_y, y2_pred, average='weighted'))
plt.subplot(1,2,1)
plt.plot(x,y2)
plt.title("F1 Score as a funtion of C(Regularization Parameter)) for training data")
plt.xlabel("C")
plt.ylabel("F1 Score")

plt.subplot(1,2,2)
plt.plot(x,y)
plt.title("F1 Score as a funtion of C(Regularization Parameter)) for test data")
plt.xlabel("C")
plt.ylabel("F1 Score")
plt.show()
plt.plot(x,y2)
plt.title("F1 Score as a funtion of C(Regularization Parameter)) for training data")
plt.xlabel("C")
plt.ylabel("F1 Score")
plt.show()

plt.plot(x,y)
plt.title("F1 Score as a funtion of C(Regularization Parameter)) for test data")
plt.xlabel("C")
plt.ylabel("F1 Score")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
x=[]
y=[]
for i in range(2,50,5):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(htrain_pca,train_y)
    y_pred = neigh.predict(htest_pca)
    x.append(i)
    y.append(metrics.f1_score(test_y, y_pred, average='weighted'))
plt.plot(x,y)
plt.title("F1 Score as a funtion of K(No of Neighbors)")
plt.xlabel("K")
plt.ylabel("F1 Score")
plt.show()
m=max(y)
index = y.index(m)
print("Max value of F1 Score occurs at K = ",x[index])
print("The Evaluation metrics are : ")
neigh = KNeighborsClassifier(n_neighbors=x[index])
neigh.fit(htrain_pca,train_y)
y_pred = neigh.predict(htest_pca)
print("Accuracy:",metrics.accuracy_score(test_y, y_pred))
print("F1 Score:",metrics.f1_score(test_y, y_pred, average='weighted'))
print("Precision: ",metrics.precision_score(test_y,y_pred, average='weighted'))
print("Recall: ",metrics.recall_score(test_y,y_pred,average = 'weighted'))
