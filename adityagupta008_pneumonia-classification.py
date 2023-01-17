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
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray')
from PIL import Image

train_images = []
train_labels = []
mean_image = np.zeros((150, 150, 3), dtype= 'float32')
total_train_files = 0

train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
sub_dirs = ['NORMAL', 'PNEUMONIA']


for sub_dir in sub_dirs:
    class_path = os.path.join(train_path, sub_dir)
    files = os.listdir(class_path)
    total_train_files += len(files)
    for file in files:
        img = Image.open(os.path.join(class_path, file))
        img = img.resize((150, 150))
        if img.mode == 'L':
            img = np.dstack([img, img, img])
        img = np.array(img)
        img = img/255
        mean_image += img
        train_images.append(img)
        label = 1 if sub_dir == 'PNEUMONIA' else 0
        train_labels.append(label)
        
mean_image = mean_image/total_train_files
import matplotlib.pyplot as plt

plt.imshow(mean_image)
plt.imshow(train_images[1])
for images in train_images:
    images -= mean_image
plt.imshow(train_images[1])
train_images_flatten = []
for image in train_images:
    image = image.flatten()
    train_images_flatten.append(image)
train_images_flatten = np.array(train_images_flatten)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_images_flatten, train_labels, test_size = 0.2, random_state= 42, shuffle= True)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(n_jobs= -1, class_weight= {0: 0.6, 1: 0.4})
model.fit(np.array(X_train), np.array(y_train))
preds = model.predict(np.array(X_test))
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, confusion_matrix

roc_auc_score(np.array(y_test), preds), f1_score(np.array(y_test), preds)
fpr, tpr, threshold = roc_curve(np.array(y_test), preds)
roc_auc = auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

cm = confusion_matrix(np.array(y_test), preds)
cm
from sklearn.metrics import precision_score, recall_score, accuracy_score
precision_score(np.array(y_test), preds, average= 'binary'), recall_score(np.array(y_test), preds, average= 'binary')
cm = confusion_matrix(y_test, preds)
cm








