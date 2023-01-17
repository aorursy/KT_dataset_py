import cv2
f, axarr = plt.subplots(2,2)
img1 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1008_virus_1691.jpeg')
img2 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person100_virus_184.jpeg')
img3 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0131-0001.jpeg')
img4 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0152-0001.jpeg')
axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)

#img1.shape
#img2.shape
#img3.shape
#img4.shape
import os
import pandas as pd
! pwd
path_train = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
path_test = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
path_val = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'

os.chdir(path_train)
lists = os.listdir(path_train)
labels = []
file_lst = []

#print(lists)

for folder in lists:
    files = os.listdir(path_train +"/"+folder)
    for file in files:
      path_file = path_train + "/" + folder + "/" + file
      file_lst.append(path_file)
      labels.append(folder)

dictP_n = {"path": file_lst,
           "label_name": labels,
          "label": labels}   

data  = pd.DataFrame(dictP_n, index = None)
data = data.sample(frac=1)
data['label'] = data['label'].replace({"NORMAL": 0, "PNEUMONIA": 1 })
data.to_csv("/kaggle/working/train.csv", index =None)

# ---------------------------------------------------------------------------

os.chdir(path_test)
lists = os.listdir(path_test)
labels = []
file_lst = []

#print(lists)

for folder in lists:
    files = os.listdir(path_test +"/"+folder)
    for file in files:
      path_file = path_test + "/" + folder + "/" + file
      file_lst.append(path_file)
      labels.append(folder)

dictP_n = {"path": file_lst,
           "label_name": labels,
          "label": labels}   

data  = pd.DataFrame(dictP_n, index = None)
data = data.sample(frac=1)
data['label'] = data['label'].replace({"NORMAL": 0, "PNEUMONIA": 1 })
data.to_csv("/kaggle/working/test.csv", index =None)

# ------------------------------------------------------------------------------------

os.chdir(path_val)
lists = os.listdir(path_val)
labels = []
file_lst = []

#print(lists)

for folder in lists:
    files = os.listdir(path_val +"/"+folder)
    for file in files:
      path_file = path_val + "/" + folder + "/" + file
      file_lst.append(path_file)
      labels.append(folder)

dictP_n = {"path": file_lst,
           "label_name": labels,
          "label": labels}   

data  = pd.DataFrame(dictP_n, index = None)
data = data.sample(frac=1)
data['label'] = data['label'].replace({"NORMAL": 0, "PNEUMONIA": 1 })
data.to_csv("/kaggle/working/val.csv", index =None)
data_train = pd.read_csv('/kaggle/working/train.csv')
data_train.tail(10)
data_train.head(10)
data_test = pd.read_csv('/kaggle/working/test.csv')
data_test.tail(10)
data_test.tail(10)
data_val = pd.read_csv('/kaggle/working/val.csv')
data_val.shape
data_val.size
data_val.head(10)
data_val.tail(10)
! pwd
os.chdir('/kaggle/working')
! pwd
! git clone https://github.com/tueimage/SE2CNN.git
! pip install tensorflow==1.13.1
# Import tensorflow and numpy
import tensorflow as tf
import numpy as np
import math as m
import time
import glob

# For validation
from sklearn.metrics import confusion_matrix
import itertools

# For plotting
from PIL import Image
from matplotlib import pyplot as plt

# Add the library to the system path
import os,sys
se2cnn_source =  os.path.join(os.getcwd(),'/kaggle/working/SE2CNN/')
if se2cnn_source not in sys.path:
    sys.path.append(se2cnn_source)

# Import the library
import se2cnn.layers
# help(se2cnn.layers.z2_se2n)
# help(se2cnn.layers.se2n_se2n)
# help(se2cnn.layers.spatial_max_pool)
# Xavier's/He-Rang-Zhen-Sun initialization for layers that are followed ReLU
def weight_initializer(n_in, n_out):
    return tf.random_normal_initializer(mean=0.0, stddev=m.sqrt(2.0 / (n_in))
    )
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def size_of(tensor) :
    # Multiply elements one by one
    result = 1
    for x in tensor.get_shape().as_list():
         result = result * x 
    return result
data_train['path']
data_val['path']
CANCERtraindata = data_train['path']
train_data = np.array([np.array(Image.open(fname)) for fname in CANCERtraindata])

# validation data
CANCERtestdata = data_val['path']
eval_data = np.array([np.array(Image.open(fname)) for fname in CANCERtestdata])
data_train['label']
data_val['label']
train_labels = data_train['label']
eval_labels = data_val['label']
print(" Length of train_data ")
print(len(train_data))
print(" Length of eval_data ")
print(len(eval_data))

print(" Length of train_labels ")
print(len(train_labels))
print(" Length of eval_labels ")
print(len(eval_labels))

#print(' Train data ')
#print(train_data)
#print(' Test data ')
#print(eval_data)

#print(' Train labels ')
#print(train_labels)
#print(' Test labels ')
#print(eval_labels)
