
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt,matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%matplotlib is a magic function in IPython
#%matplotlib inline sets the backend of 
#matplotlib to the 'inline' backend: 
#With this backend, the output of plotting commands
#is displayed inline within frontends like the Jupyter notebook, 
#directly below the code cell that produced it.
%matplotlib inline

#constants
NUMBER_OF_TRAINING_IMGS=5000
IMG_HEIGHT=28
IMG_WIDTH=28
#list all the contents of the ../input directory
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
labeled_images=pd.read_csv('../input/train.csv')
print('shape of the dataframe ',labeled_images.shape)
labeled_images.head(n=3)
images=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,1:] # first NUMBER_OF_TRAINING_IMGS rows,column 2 onwards.
labels=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,:1] #first NUMBER_OF_TRAINING_IMGS rows, first column. 
                                                        #I could have used .iloc[0:NUMBER_OF_TRAINING_IMGS,0 ] 
                                                        #instead of        .iloc[0:NUMBER_OF_TRAINING_IMGS,:1] but the first case returns a Series and the second one a DataFrame.
                                                        #prefer the latter.
train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)
type(train_images)
ii=13
img=train_images.iloc[ii].values
print('shape of numpy array',img.shape,'reshaping to 28x28')
img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[ii,0])

plt.hist(train_images.iloc[ii].values)
train_images.describe()
max=train_images.max()
min=train_images.min()
train_images_std=(train_images-min)/(max-min)
test_images_std=(test_images-min)/(max-min)
train_images_std.head(n=2)
train_images_std=train_images_std.replace([-np.inf,np.inf],np.nan)
test_images_std=test_images_std.replace([-np.inf,np.inf],np.nan)
train_images_std=train_images_std.fillna(0)
test_images_std=test_images_std.fillna(0)
train_images_std.head(n=2)
plt.hist(train_images_std.iloc[13])
test_images_std.describe()
clf=SVC(kernel='rbf',C=1.0,random_state=1,gamma=0.1)   # radial basis function K(a,b) = exp(-gamma * ||a-b||^2 ) where gamma=1/(2*stddev)^2
clf.fit(train_images_std,train_labels.values.ravel())
clf.score(test_images_std,
          test_labels.values.ravel())

## Will not be using this classifier. This  is to test the need for scaling in SVM

clf_without_std=SVC(kernel='rbf',C=1.0,random_state=1,gamma=0.1)   # radial basis function K(a,b) = exp(-gamma * ||a-b||^2 ) where gamma=1/(2*stddev)^2
clf_without_std.fit(train_images,train_labels.values.ravel())      # unscaled original  data.
clf_without_std.score(test_images,
          test_labels.values.ravel())
#use the clf model. (trained on scaled data)
raw_data=pd.read_csv('../input/test.csv')
print('shape of dataframe ',raw_data.shape)
raw_data.head(n=3)
raw_data_scaled=(raw_data-min)/(max-min)
raw_data_scaled=raw_data_scaled.replace([-np.inf,np.inf],np.nan)
raw_data_scaled=raw_data_scaled.fillna(0)
raw_data_scaled.describe()
from random import randint

jj=13

for i in range(1,3):
    jx=randint(0,raw_data_scaled.shape[0])
    smp=raw_data_scaled.iloc[[jx]]
    img=smp.values        #get a random image
    img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
    for j in range(1,3):
        for k in range(1,2):
            print('plt',i,j,k)
            plt.subplot(i,j,k)
            plt.imshow(img)
            y=clf.predict(smp)
            plt.title('predicted : '+str(y[0]))
            

'''
img=raw_data_scaled.iloc[[jj]].values
img=img.reshape(IMG_HEIGHT,IMG_WIDTH)
img1=raw_data_scaled.iloc[jj+1].values
img1=img1.reshape(IMG_HEIGHT,IMG_WIDTH)
plt.subplot(121)
plt.imshow(img)
plt.title('predicted: ')
plt.subplot(122)
plt.imshow(img1)
plt.title('predicted: ')
'''
jjthSample=raw_data_scaled.iloc[[jj]]     # iloc[j] returns a Series. We need a dataframe to pass to predict. which is returned by iloc[[jj]]  .i.e. a list of rows
type(jjthSample)
y_pred_jj=clf.predict(jjthSample)
y_pred_jj
y_pred=clf.predict(raw_data_scaled)
y_pred.shape
submissions=pd.DataFrame({"ImageId":list(range(1,len(y_pred)+1)), "Label":y_pred})
submissions.head()
submissions.to_csv("mnist_svm_submit.csv",index=False,header=True)
!ls
