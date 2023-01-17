%matplotlib inline
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import os 
 
from sklearn.metrics import accuracy_score 
import tensorflow as tf 
import keras
df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
df1 = df.copy()
df1.drop('label', axis=1,inplace=True)
labels = df['label'].values
img_array = df1.values
img_array = img_array.reshape(-1,28,28)
img_array.shape
seed = 128 
rng = np.random.RandomState(seed)
#labels = keras.utils.to_categorical(labels)
split_size = int(img_array.shape[0]*0.7) 
train_x, val_x = img_array[:split_size], img_array[split_size:] 
train_y, val_y = labels[:split_size], labels[split_size:]
img = img_array[1]
plt.imshow(img)
img.shape
from skimage.feature import hog
from skimage.transform import resize
resized_img = resize(img, (128,64,1))
ppc = 16
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
plt.imshow(hog_image)
hog_images = []
hog_feature = []
for i in range(len(img_array)):
    x = img_array[i]
    resized_img = resize(x, (128,64,1))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_images.append(hog_image)
    hog_feature.append(fd)
hog_feature = np.array(hog_feature)
from sklearn import svm
clf = svm.SVC()
train = pd.DataFrame(hog_feature)
target = pd.DataFrame(labels)
clf.fit(train, target)
pred = clf.predict(train)
from sklearn.metrics import classification_report,accuracy_score
print("Accuracy: "+str(accuracy_score(pred, target)))
print('\n')
print(classification_report(pred, target))
pred
test = pd.read_csv('../input/digit-recognizer/test.csv')
test = test.values
test = test.reshape(-1,28,28)
test_hog_images = []
test_hog_feature = []
for i in range(len(test)):
    x = test[i]
    resized_img = resize(x, (128,64,1))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    test_hog_images.append(hog_image)
    test_hog_feature.append(fd)
test_hog_feature = np.array(test_hog_feature)
test = pd.DataFrame(test_hog_feature)

pred_test=clf.predict(test)
#label = np.argmax(pred, axis=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = pred_test
submission.to_csv('submission.csv', index=False)
submission
