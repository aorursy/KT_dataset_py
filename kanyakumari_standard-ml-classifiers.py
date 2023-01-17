import h5py 

import numpy as np

import matplotlib.pyplot as plt

from skimage.color import rgb2gray

from sklearn.cluster import KMeans

from skimage import filters

from skimage import exposure

import cv2

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
with h5py.File('../input/all_mias_scans.h5', 'r') as scan_h5:

    bg_info = scan_h5['BG'][:]

    class_info = scan_h5['CLASS'][:]

    # low res scans

    scan_lr = scan_h5['scan'][:][:, ::16, ::16]
scan_lr_flat2 = scan_lr.reshape((scan_lr.shape[0], -1))

scan_lr_flat2 = cv2.medianBlur(scan_lr_flat2, 9)

#scan_lr_flat1 = cv2.medianBlur(scan_lr, 9)

#figure_size=7

#scan_lr_flat2 = cv2.GaussianBlur(scan_lr_flat1, (figure_size, figure_size),0)

#scan_lr_flat2 = cv2.GaussianBlur(scan_lr, (figure_size, figure_size),0)

#scan_lr_flat2 = cv2.Laplacian(scan_lr_flat1,cv2.CV_64F)



# region based segmentation



#gray=scan_lr_flat2

#gray_r = gray.reshape(gray.shape[0]*gray.shape[1])

#for i in range(gray_r.shape[0]):

 #   if gray_r[i] > gray_r.mean():

  #      gray_r[i] = 1

   # else:

    #    gray_r[i] = 0

#gray = gray_r.reshape(gray.shape[0],gray.shape[1])

#pic=scan_lr_flat2

#pic_n = pic.reshape(pic.shape[0]*pic.shape[1])



#kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)

#pic2show = kmeans.cluster_centers_[kmeans.labels_]

#scan_lr_flat2 = filters.sobel_h(scan_lr_flat2)

# segmentation using otsu method



#from skimage import data





ret,th1 = cv2.threshold(scan_lr_flat2,127,255,cv2.THRESH_BINARY)

#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

for i in xrange(4):

    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')

    plt.title(titles[i])

    plt.xticks([]),plt.yticks([])

plt.show()

#val = filters.threshold_otsu(scan_lr_flat2)



#hist, bins_center = exposure.histogram(scan_lr_flat2)



#plt.figure(figsize=(9, 4))

#plt.subplot(131)

#plt.imshow(camera, cmap='gray', interpolation='nearest')

#plt.axis('off')

#plt.subplot(132)

#plt.imshow(camera < val, cmap='gray', interpolation='nearest')

#plt.axis('off')

#plt.subplot(133)

#plt.plot(bins_center, hist, lw=2)

#plt.axvline(val, color='k', ls='--')



#plt.tight_layout()
scan_lr_flat2.shape
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

class_le.fit(class_info)

class_vec = class_le.transform(class_info)

class_le.classes_
from sklearn.feature_extraction import image

deneme=image.extract_patches_2d(scan_lr_flat2[0],(3,3))

deneme
plt.imshow(scan_lr_flat2[14],cmap="gray")
scan_lr_flat2[14].shape
set(deneme.flatten())
class_info.shape
from skimage.feature import local_binary_pattern

lbp = local_binary_pattern(scan_lr_flat2[14], 16, 4, "uniform")
set(lbp.flatten())
lbp.shape
lbp.ravel().shape
set(lbp.ravel())
sns.countplot(x=lbp.ravel())
features=[]

for img in scan_lr_flat2:

    lbp = local_binary_pattern(img, 16, 4, "uniform")

    flatten=lbp.ravel()

    features.append(flatten)
x=pd.DataFrame(features)

x.head(5)

len(x)
from sklearn.decomposition import PCA

pca=PCA(n_components=10)

transx=pca.fit_transform(x)
transx.shape
X=transx

y=class_info
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1945)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

scalex=sc.fit_transform(X)
scalex=pd.DataFrame(scalex)

scalex.head(5)
from sklearn.model_selection import train_test_split

idx_vec = np.arange(scan_lr_flat2.shape[0])

x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(scan_lr_flat2, 

                                                    class_vec, 

                                                    idx_vec,

                                                    random_state = 2017,

                                                   test_size = 0.3,

                                                   stratify = class_vec)

#from sklearn.preprocessing import StandardScaler



#sc = StandardScaler()

#x_train = sc.fit_transform(x_train)

#x_test = sc.transform(x_test)

print('Training', x_train.shape)

print('Testing', x_test.shape)



#plt.imshow(scan_lr_flat[0], cmap = 'bone')
# useful tools

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

creport = lambda gt_vec,pred_vec: classification_report(gt_vec, pred_vec, 

                                                        target_names = [x.decode() for x in 

                                                                        class_le.classes_])
from sklearn.dummy import DummyClassifier

dc = DummyClassifier(strategy='most_frequent')

dc.fit(x_train, y_train)

y_pred = dc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
#ANN model

#from keras.models import Sequential

#from keras.layers import Dense



#classifier = Sequential() # Initialising the ANN



#classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

#classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'
#classifier.fit(x_train, y_train, batch_size = 1, epochs = 100)
%%time

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(8)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))



#svm=KNeighborsClassifier(n_neighbors=9)

#svm.fit(x_train,y_train)

#ypred=svm.predict(x_test)

#from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#print(confusion_matrix(y_pred=ypred,y_true=y_test))

#print(accuracy_score(y_pred=ypred,y_true=y_test))

#print(classification_report(y_pred=ypred,y_true=y_test))
%%time

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
%%time

from xgboost import XGBClassifier

xgc = XGBClassifier(silent = False, nthread=2)

xgc.fit(x_train, y_train)

y_pred = xgc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
from tpot import TPOTClassifier

tpc = TPOTClassifier(generations = 2, population_size=5, verbosity=True)

tpc.fit(x_train, y_train)

y_pred = tpc.predict(x_test)

print('Accuracy %2.2f%%' % (100*accuracy_score(y_test, y_pred)))

print(creport(y_test, y_pred))
fig, ax1 = plt.subplots(1,1)

ax1.matshow(np.log10(confusion_matrix(y_test, y_pred).clip(0.5,1e9)), cmap = 'RdBu')

ax1.set_xticks(range(len(class_le.classes_)))

ax1.set_xticklabels([x.decode() for x in class_le.classes_])

ax1.set_yticks(range(len(class_le.classes_)))

ax1.set_yticklabels([x.decode() for x in class_le.classes_])

ax1.set_xlabel('Predicted Class')

ax1.set_ylabel('Actual Class')
fig, c_axs = plt.subplots(3,3, figsize = (12,12))

for c_ax, test_idx in zip(c_axs.flatten(), np.where(y_pred!=y_test)[0]):

    c_idx = idx_test[test_idx]

    c_ax.imshow(scan_lr[c_idx], cmap = 'bone')

    c_ax.set_title('Predict: %s\nActual: %s' % (class_le.classes_[y_pred[test_idx]].decode(),

                                               class_le.classes_[y_test[test_idx]].decode()))

    c_ax.axis('off')

# segmetation using otsu method

import matplotlib.pyplot as plt

from skimage import data

from skimage import filters

from skimage import exposure



camera = data.camera()

val = filters.threshold_otsu(camera)



hist, bins_center = exposure.histogram(camera)



plt.figure(figsize=(9, 4))

plt.subplot(131)

plt.imshow(camera, cmap='gray', interpolation='nearest')

plt.axis('off')

plt.subplot(132)

plt.imshow(camera < val, cmap='gray', interpolation='nearest')

plt.axis('off')

plt.subplot(133)

plt.plot(bins_center, hist, lw=2)

plt.axvline(val, color='k', ls='--')



plt.tight_layout()

plt.show()
import h5py



#filename = "vstoxx_data_31032014.h5"



#h5 = h5py.File(filename,'r')

with h5py.File('../input/all_mias_scans.h5', 'r') as h5:

    futures_data = h5['futures_data']  # VSTOXX futures data

    options_data = h5['options_data']  # VSTOXX call option data

print(futures_data)

h5.close()