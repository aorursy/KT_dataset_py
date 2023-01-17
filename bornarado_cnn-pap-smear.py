# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
from PIL import Image
import sklearn
print(os.listdir("/kaggle/input/papsmeardatasets/"))
Herlev = "/kaggle/input/papsmeardatasets/herlev_pap_smear/"
Sipak = "/kaggle/input/papsmeardatasets/sipakmed_fci_pap_smear/"
Lista_H = os.listdir(Herlev)
def get_label(Dir):
    for nextDir in os.listdir(Dir):
            if nextDir.startswith('normal') or nextDir.startswith('benign'):
                label = 0
            elif nextDir.startswith('abnormal'):
                label = 1
            
    return nextDir, label
def get_data(Dir):
    X = []
    y = []
    
    for dir_name in Dir:
        for nextDir in os.listdir(dir_name):
            if not nextDir.startswith('.'):
                if nextDir.startswith('normal'):
                    label = 0
                elif nextDir.startswith('abnormal') or nextDir.startswith('benign'):
                    label = 1
                


                temp = dir_name + nextDir

                for file in tqdm(os.listdir(temp)):
                    if not file.endswith('d.bmp') and not file.endswith('dat'):
                        img = cv2.imread(temp + '/' + file)
                        if img is not None:
                            img = skimage.transform.resize(img, (64,64,3))
                            #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                            img = np.asarray(img)
                            X.append(img)
                            y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
Lista = [Herlev, Sipak]
X_train, y_train = get_data(Lista)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15)
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =["accuracy",tf.keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)])
#metrics =[ tf.keras.metrics.Recall()]
plot_model(classifier, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
display(Image.open('cnn_model.png'))



classifier.fit(X_train, y_train,validation_split = 0.1,  epochs=20)

from matplotlib import pyplot
import matplotlib.pyplot as plt
print(classifier.history.history.keys())
plt.plot(classifier.history.history['accuracy'])
plt.plot(classifier.history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#history1.history['val_loss']
# Plot training & validation loss values
plt.plot(classifier.history.history['loss'])
plt.plot(classifier.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
classifier.save('Cell_predict')
new_model = tf.keras.models.load_model('Cell_predict')
predictions = new_model.predict([X_test])
niz = np.arange(0.0, 1.01, 0.01)
def zaok(a, x):
    if a >= x:
        return 1
    else:
        return 0
y_test[0] == np.rint(predictions[0])
TP = 0
FP = 0
TN = 0
FN = 0
for (i, j) in zip (y_test, predictions):
    if i == 0:
        if i == np.rint(j):
            TN += 1
        else:
            FP += 1
    elif i == 1:
        if i == np.rint(j):
            TP += 1
        else:
            FN += 1
     
            
print(TP, FP, TN, FN)
def prec(TP,FP):
    return TP/(TP+FP)
def rec(TP, FN):
    return TP/(TP+FN)
precision = []
recall = []
pom_pred = []

for k in niz:
    TP_1 = 0
    FP_1 = 0
    TN_1 = 0
    FN_1 = 0
    for (i, j) in zip (y_test, predictions):
        if i == 0:
            if i == zaok(j, k):
                TN_1 += 1
            else:
                FP_1 += 1
        elif i == 1:
            if i == zaok(j, k):
                TP_1 += 1
            else:
                FN_1 += 1
    precision.append(prec(TP_1,FP_1))
    recall.append(rec(TP_1,FN_1))
    
    
    
brojac = 0

for (i, j) in zip (y_test, predictions):
    if i != np.rint(j):
        brojac += 1
    
accu = 1 - brojac/len(y_test)
print('Accuracy je:', accu)
Rec_ = TP/(FN+TP)
print(Rec_)
Pre_ = TP/(FP+TP)
print(Pre_)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.annotate('Najbolji Recall i precision', xy=(Rec_, Pre_), xytext=(1, 1.1),
            arrowprops=dict(facecolor='black', shrink=0.002),
            )
y_pred = []
for i in predictions:
    y_pred.append(np.rint(i))

arr = np.rint(predictions)
y_pred2 = []
for i in y_pred:
    y_pred2.append(i[0])
cm = sklearn.metrics.confusion_matrix(y_test,arr)
from sklearn.metrics import plot_confusion_matrix
cm
import matplotlib.pyplot as plt
import seaborn as sns

cn = sns.light_palette("blue", as_cmap=True)
x=pd.DataFrame(cm)
x=x.style.background_gradient(cmap=cn)
display(x)
%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools

plt.imshow(cm, interpolation='nearest', cmap=cn)
plt.title("Matrica konfuzije")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['P', 'N'])

#if normalize:
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    print("Normalized confusion matrix")
#else:
#    print('Confusion matrix, without normalization')

#print(cm)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
