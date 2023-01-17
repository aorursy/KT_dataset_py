import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from sklearn.decomposition import PCA
import math 
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img,SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def show_digits(lst,features,labels):
    bn=np.array(lst)

    bnm=bn.reshape(math.ceil(len(bn)/2),2)
    x = np.asarray(features)
    y = np.asarray(labels)
    n_col = 8.0
    n_row = np.ceil(len(lst)/n_col/2 )
    
    n_col=int(n_col)
    n_row = int(n_row)
    
    aspect = 1
    n = n_row # number of rows
    m = n_col # numberof columns
    bottom = 0.1; left=0.05
    top=1.-bottom; right = 1.-left
    fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
    #widthspace, relative to subplot size
    wspace=0.15  # set to zero for no spacing
    hspace=wspace/float(aspect)*3
    #fix the figure height
    figheight= 1*n_row # inch
    figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
    
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                        wspace=wspace, hspace=hspace)
    
    i=0
    for ax in axes.flatten():
        if i < len(bnm):
            ax.imshow(x[bnm[i][0],:].reshape(56,56))
            ax.set_title(str(y[bnm[i][0]])+' as '+str(bnm[i][1]),fontsize=8, y=0.97)
            ax.axis('off')
            i+=1
        else: break
    plt.show()

def prep_image(pict,oldSize,newSize):
    #raw = np.asarray(pict)
    image = pict.reshape(oldSize,oldSize)
    rsimage = cv2.resize(image, (0,0), fx=2, fy=2)     
    skewed_image = deskew(rsimage,newSize)
    gray = cv2.GaussianBlur(skewed_image, (3, 3), 0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,0)  
    return thresh.flatten()
t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:10000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))
data[data>0]=1
from sklearn.cross_validation import train_test_split
t0 = time()
train_images,test_images,train_labels,test_labels=train_test_split(data,labels,train_size=0.8,random_state=0,stratify=labels)

len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))
print (train_images.shape)
print (test_images.shape)
print (train_labels.shape)
print (test_labels.shape)
print (len_test_labels)
print (len_train_labels)
merged_labels=np.concatenate((train_labels,test_labels),axis=0)
ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels1= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]
testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)
t0 = time()

clf = OneVsRestClassifier(svm.LinearSVC())
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print ("Linear SVC done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
t0 = time()

clf = OneVsRestClassifier(svm.SVC(kernel='rbf'))
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print (" SVC with 'rbf' kernel done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
t0 = time()

clf = OneVsRestClassifier(svm.SVC(kernel='poly'))
clf.fit(train_images, train_labels1)
acc= clf.score(test_images,test_labels1)  

print (" SVC with 'poly' kernel done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
from sklearn.decomposition import PCA

t0 = time()
n_s = np.linspace(0.65, 0.85, num=14)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf'))
    clf.fit(train_images_new, train_labels1)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))
%matplotlib inline
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)
from sklearn.decomposition import PCA

t0 = time()
n_s = np.linspace(0.65, 0.85, num=21)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced'))
    clf.fit(train_images_new, train_labels1)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))
%matplotlib inline
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)
print ("PCA starts...")
t0 = time()
max_n_components=29
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))
param_grid = {
    "estimator__C": [.1, 1, 10, 100, 1000],
    "estimator__gamma":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)
param_grid = {
    "estimator__C": [1,3,7,9, 10,11,13,16,19,22,25,30],
    "estimator__gamma":[0.005,0.007,0.008,0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)
t0 = time()
print ("read data,skew, blur and threshold done in...")
data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

data[data>0]=1

from sklearn.cross_validation import train_test_split
t0 = time()
train_images,test_images,train_labels,test_labels=train_test_split(data,labels,train_size=0.8,random_state=0,stratify=labels)

len_test_labels=len(test_labels)
len_train_labels=len(train_labels)
print ("read data skew, blur and threshold done in : %0.3fs" % (time() - t0))
merged_labels=np.concatenate((train_labels,test_labels),axis=0)
ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels1= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]

print ("PCA starts...")
t0 = time()
max_n_components=29
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))

testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced',C=13,gamma=0.015))
clf.fit(train_images_new, train_labels1)
acc= clf.score(test_images_new,test_labels1)  
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
%matplotlib inline
data = shuffle(pd.read_csv('../input/train.csv'))[0:5000]

features=np.array(data.ix[:,1:])
labels = np.array(data.ix[:,0]) 

x = np.array(data,dtype = 'uint8') 

samples=[]
for i in range(0,10):
    samples.append( x[x[:,0] ==i][0:10])

sa =[]
sa = np.vstack(samples)

x=features[0:40]
n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sa):
        ax.imshow(sa[i,1:].reshape(28,28), cmap='gray_r')
        ax.set_title(str(sa[i,0]),fontsize=8, y=0.97)
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()

sar=sa
sar[:,1:][sar[:,1:]>0]=1

x=features[0:40]

n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sar):
        ax.imshow(sar[i,1:].reshape(28,28), cmap='gray_r')
        ax.set_title(str(sar[i,0]),fontsize=8, y=0.97)
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img,SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def prep_image(pict,oldSize,newSize):
    raw = np.asarray(pict)
    image = raw.reshape(oldSize,oldSize)
    rsimage = cv2.resize(image, (0,0), fx=2, fy=2)     
    skewed_image = deskew(rsimage,newSize)
    gray = cv2.GaussianBlur(skewed_image, (3, 3), 0)
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,0)  
    return thresh.flatten()
x = np.array(data,dtype = 'uint8') 

samples=[]
for i in range(0,10):
    samples.append( x[x[:,0] ==i][0:10])

sa =[]
sa = np.vstack(samples)


newSize=56 #size after resizing, from 28 to 56
newdata_imageProcessed = []
for row in sa[:,1:]:     
     temp =prep_image(row,28,newSize)
     newdata_imageProcessed.append(temp)

sas = np.array(newdata_imageProcessed ).astype(int)
sas[sas>0]=1


x=features[0:40]

n_col=10
n_row = 10

aspect = 1
print (aspect)
n = n_row # number of rows
m = n_col # numberof columns
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-left
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
#widthspace, relative to subplot size
wspace=0.15  # set to zero for no spacing
hspace=wspace/float(aspect)*3
#fix the figure height
figheight= 1.5*n_row # inch
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp


fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

i=0
for ax in axes.flatten():
    if i < len(sas):
        ax.imshow(sas[i,:].reshape(56,56), cmap='gray_r')
        
        #ax.axis('off')
        ax.spines['bottom'].set_color(None)
        ax.spines['top'].set_color(None)
        ax.spines['right'].set_color(None)
        ax.spines['left'].set_color(None)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        i+=1
    else: break

plt.show()
t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:10000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))



train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)
t0 = time()
n_s = np.linspace(0.65, 0.85, num=14)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', class_weight='balanced'))
    clf.fit(train_images_new, train_labels)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
    print(n)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))
%matplotlib inline
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)
t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv')[0:20000] # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))



train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)
t0 = time()
n_s = np.linspace(0.65, 0.85, num=11)
accuracy = []
for n in n_s:
    pca = PCA(n_components=n)
    pca.fit(train_images)
    train_images_new = pca.transform(train_images)
    test_images_new = pca.transform(test_images)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', class_weight='balanced'))
    clf.fit(train_images_new, train_labels)
    acc= clf.score(test_images_new,test_labels1)   
    accuracy.append(acc)
    accuracy.append(pca.n_components_)
    print("done Iteration for: ",n)
print ("PCA iterations with SVM (kernel:rbf) takes[sec]: %0.3fs" % (time() - t0))
%matplotlib inline
import matplotlib.pyplot as plt
accuracyarr = np.array(accuracy) 
accuracyarr=accuracyarr.reshape(math.ceil(len(accuracyarr)/2),2)
plt.plot(accuracyarr[:,1],accuracyarr[:,0], 'b-')
print (accuracyarr)
print ("PCA starts...")
t0 = time()
max_n_components=150
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))

t0 = time()
#n_estimators=10
print ("modelling starts...")
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced'))
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
param_grid = {
    "estimator__C": [5, 10, 50],
    "estimator__gamma":[ 0.001, 0.005, 0.01,0.05]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)
param_grid = {
    "estimator__C": [8, 10, 15,20],
    "estimator__gamma":[ 0.003, 0.005, 0.007]
}
clf = GridSearchCV(OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced')), param_grid)
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
print (" Gridsearch done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)
print (clf.best_params_)
t0 = time()
print ("read data,resize, skew, blur and threshold it...")
data = pd.read_csv('../input/train.csv') # Read csv file in pandas dataframe
labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

x = np.array(data,dtype = 'uint8') 
SZ=56
newdata_imageProcessed = []
for row in x:     
     temp =prep_image(row,28,SZ)
     newdata_imageProcessed.append(temp)

newdata = np.array(newdata_imageProcessed ).astype(int)
newdata[newdata>0]=1
print ("read data, skew, blur and threshold done in [sec]: %0.3fs" % (time() - t0))


t0 = time()
train_images, test_images,train_labels, test_labels = train_test_split(newdata, labels, train_size=0.8, random_state=0,stratify=labels)
len_test_labels=len(test_labels)
len_train_labels=len(train_labels)

merged_labels=np.concatenate((train_labels,test_labels),axis=0)

ohe = OneHotEncoder()
labels3 = np.array(ohe.fit_transform(merged_labels.reshape(-1, 1)).todense(),dtype = 'uint8') 
train_labels= labels3[:len_train_labels,:]
test_labels1= labels3[len_train_labels:,:]


testlabels_concat=np.append(test_labels1,test_labels[:, np.newaxis], axis=1)
testlabels_concat_unique=unique_rows(testlabels_concat)


print ("train-test split and one hot encoding done in [sec]: %0.3fs" % (time() - t0))
print ("PCA starts...")
t0 = time()
max_n_components=150
pca = PCA(n_components=max_n_components)
pca.fit(train_images)
train_images_new = pca.transform(train_images)
test_images_new = pca.transform(test_images)
print ("PCA done in %0.3fs" % (time() - t0))
print ("Classifiying and fitting starts...")
t0 = time()
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',class_weight='balanced',C=10,gamma=0.005))
clf.fit(train_images_new, train_labels)
acc= clf.score(test_images_new,test_labels1)  
predicted_test_labels1=clf.predict(test_images_new)
print (" Classifiying and fitting done in: %0.3fs" % (time() - t0))
print ("accuracy:",acc)