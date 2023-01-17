# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
# Loading the data
df_train=pd.read_csv("../input/train.csv")
# Looking at the shape of the data
df_train.shape
# Function to first calculate moments of the image data, which is the first step to deskewing the image
from scipy.ndimage import interpolation

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix
# Function used for deskewing the image which internally first calls the moment function described above
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)
# Function for scaling the data between 0 and 1
def scale(vect):
    return (vect-vect.min())/(vect.max()-vect.min())
# Getting the features from the train data frame by dropping label from the data frame
df_X=df_train.drop("label",axis=1)
# Deskewing the data
df_X=df_X.apply(lambda x: deskew(x.reshape(28,28)).flatten(),axis=1)
# Scaling the data
X=df_X.apply(scale)
# Dropping all the columns with only NaN values
X=X.dropna(axis=1,how='all')
# Saving the label data as the target variable
y=df_train["label"]
# Observing first few lines of the features data frame
X.head()
# Splitting the data into test and training set for our first simple linear SVM testing
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
# Creating our linear SVM object
from sklearn.svm import SVC
clf=SVC(C=1,kernel="linear")
# Fitting the training data in the SVM object declaring before
clf.fit(train_x,train_y)
# Saving the predictions on the test set 
y_predict=clf.predict(test_x)
# Measuring the accuracy of our predictions
from sklearn import metrics
accuracy=metrics.accuracy_score(test_y,y_predict)
print(accuracy)
# A function where we going to test different level of PCA decomposition and see how it improves accuracy
from sklearn.decomposition import PCA
def n_component_analysis(n,X_train, y_train, X_val, y_val,kernel_type="linear"):
   
    pca = PCA(n_components=n)
    print("PCA begin with n_components: {}".format(n))
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    print('SVC begin')
    clf1 = SVC(kernel=kernel_type)
    clf1.fit(X_train_pca, y_train)
    predictions=clf1.predict(X_val_pca)
    accuracy = metrics.accuracy_score(y_val,predictions)
    
    print("accuracy: {}".format(accuracy))
    return accuracy
# Calling the PCA function above where kernel is linear by default and see if we ca improve our accuracy
n_s = np.linspace(0.70, 0.85, num=15)
accuracy = []
for n in n_s:
    tmp = n_component_analysis(n,train_x, train_y, test_x, test_y)
    accuracy.append(tmp)
del accuracy
accuracy = []
for n in n_s:
    tmp = n_component_analysis(n,train_x, train_y, test_x, test_y,"rbf")
    accuracy.append(tmp)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(n_s,np.array(accuracy),label='Accuracy vs % variance explained')
plt.legend()
plt.show()
# Doing 4 fold cross validation for hyper parameter tuning
#from sklearn.model_selection import GridSearchCV
#n_fold=4
#param={'C':[1,10,20],'gamma':[0.0001,0.001,0.01]}
#model=SVC(kernel='rbf')

#clf2=GridSearchCV(estimator=model,param_grid=param,cv=n_fold,verbose=1,return_train_score=True,scoring="accuracy")
#clf2.fit(X,y)
# Printing results from Gridsearch
#cv_results = pd.DataFrame(clf2.cv_results_)
#cv_results.sort_values('mean_test_score',ascending=False)
# Do the PCA transform with n=0.83 and after that fitting a 'rbf' kernel with C=20 and gamma=0.001
pca = PCA(n_components=0.83)
pca.fit(train_x)
X_train_pca = pca.transform(train_x)
X_val_pca = pca.transform(test_x)

clf1 = SVC(kernel="rbf",C=20,gamma=0.01)
clf1.fit(X_train_pca,train_y)
predictions=clf1.predict(X_val_pca)
metrics.accuracy_score(test_y,predictions)
# Loading the test data
df_test=pd.read_csv('../input/test.csv')
# Deskewing the data
df_test=df_test.apply(lambda x:deskew(x.reshape(28,28)).flatten(),axis=1)
# Scaling the data
df_test=(df_test-df_X.min())/(df_X.max()-df_X.min())
# Only looking at the features that were present in our training set as well
X_t=df_test[X.columns]
# Doing the PCA transform
X_trans=pca.transform(X_t)
# Doing the predictions and saving in a dataframe
final=clf1.predict(X_trans)
df=pd.DataFrame(final,columns=['Label'],index=np.arange(1,28001))
# Saving the dataframe into a result.csv file
df.to_csv('subimission.csv',index_label='ImageId')
