# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from matplotlib import pyplot as plt
import os
from subprocess import check_output
import cv2
from PIL import Image
import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#print(check_output(["ls","../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
trainLabels = pd.read_csv("../input/diabetic-retinopathy-detection/trainLabels.csv")
trainLabels.head()
filelist = glob.glob('../input/diabetic-retinopathy-detection/*.jpeg') 
np.size(filelist)
# Load, resize and save the image data
img_data = []
img_label = []
img_r = 224
img_c = 224
for file in filelist:
    tmp = cv2.imread(file)
    tmp = cv2.resize(tmp,(img_r, img_c), interpolation = cv2.INTER_CUBIC)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    tmp = cv2.normalize(tmp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_data.append(np.array(tmp).flatten())
    tmpfn = file
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    img_label.append(trainLabels.loc[trainLabels.image==tmpfn, 'level'].values[0])
#import pickle
#with open('../diabetic-retinopathy-detection/img_data', 'wb') as f:
#    pickle.dump(img_data, f)
print(len(img_data))
print(len(img_label))
fileName = []
eye = []
for file in filelist:
    tmpfn = file
    tmpfn = tmpfn.replace("../input/diabetic-retinopathy-detection/","")
    tmpfn = tmpfn.replace(".jpeg","")
    fileName.append(tmpfn)
    if "left" in tmpfn:
        eye.append(1)
    else:
        eye.append(0)
#data = pd.DataFrame({'fileName':fileName,'eye':eye,'img_data':img_data,'label':img_label}) # keyerror 10
data = pd.DataFrame({'eye':eye,'img_data':img_data,'label':img_label})
data.sample(3)
data[['eye','label']].hist(figsize = (10, 5))
from sklearn.model_selection import train_test_split
X = data['img_data']
y = data['label']
#X = np.asarray(img_data)
#y = np.asarray(img_label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.utils import shuffle

data,label = shuffle(X_train,y_train, random_state=2)
train_data = pd.DataFrame({'data': data, 'label':label})
train_df = train_data.groupby(['label']).apply(lambda x: x.sample(160, replace = True)
                                                      ).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', train_data.shape[0])
train_df[['label']].hist(figsize = (10, 5))
#train_df = data
#train_df[['label', 'eye']].hist(figsize = (10, 5))
from sklearn.model_selection import train_test_split
X_train = train_df['data']
y_train = train_df['label']
#X = np.asarray(img_data)
#y = np.asarray(img_label)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
print(X_train.shape)
print(y_train.shape)
# Reshaping the Training data for model input
#print(type(X_train)) # <class 'pandas.core.series.Series'>
#print(X_train.shape) # (800,)
X_train_resh = np.zeros([X_train.shape[0],img_r, img_c, 1])
for i in range (X_train.shape[0]-1):
    X_train_resh[i] = np.reshape(X_train[i], (img_r, img_c, 1))
    #print(X_train_resh.shape) # (800,img_r,img_c,3)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_test_resh = np.zeros([X_test.shape[0],img_r, img_c, 1])
for i in range (X_test.shape[0]-1):
    X_test_resh[i] = np.reshape(X_test[i], (img_r, img_c, 1))
print(X_test_resh.shape) # (800,img_r,img_c,3)
from keras.utils import np_utils
# convert class vectors to binary class matrices
nb_classes = 5
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#from sklearn.utils import shuffle
#from scipy.sparse import coo_matrix
#X_sparse = coo_matrix(X_train_resh)
#X_train_resh, X_sparse, Y_train = shuffle(X_train_resh, X_sparse, Y_train, random_state=0)
import matplotlib.pyplot as plt
import matplotlib
#plt.imshow(X_train_resh[100])

img=X_train_resh[100].reshape(img_r,img_c)
plt.imshow(img)
plt.imshow(img,cmap='gray')
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
# Create CNN model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape = (img_r,img_c,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Dropout(0.75))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Calculate the class weights for unbalanced data
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weight = compute_class_weight("balanced", classes, y_train)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# convert class vectors to binary class matrices
nb_classes = 5
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# Fit the model
model.fit(X_train_resh, Y_train, batch_size = 32, epochs=30, verbose=1,class_weight=class_weight)
score = model.evaluate(X_test_resh, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

from keras.models import model_from_json
# Model to JSON
model_json = model.to_json()
with open("model_project_work.json", "w") as json_file:
    json_file.write(model_json)
# Weights to HDF5
model.save("model_project_work.h5")
print("Saved model to disk")
# Load saved model
from keras.models import load_model
from keras.models import Sequential
model = load_model('model_project_work.h5')
from sklearn.metrics import accuracy_score, classification_report
pred_Y = model.predict(X_test_resh, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(Y_test, -1)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat))

import seaborn as sns
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat), 
            annot=True, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = X_test_resh.shape[0]//16)
from sklearn.metrics import roc_curve, roc_auc_score
sick_vec = test_Y_cat>0
sick_score = np.sum(pred_Y[:,1:],1)
fpr, tpr, _ = roc_curve(sick_vec, sick_score)
fig, ax1 = plt.subplots(1,1, figsize = (6, 6), dpi = 150)
ax1.plot(fpr, tpr, 'b.-', label = 'Model Prediction (AUC: %2.2f)' % roc_auc_score(sick_vec, sick_score))
ax1.plot(fpr, fpr, 'g-', label = 'Random Guessing')
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');