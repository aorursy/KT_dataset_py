# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
import glob

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
%matplotlib inline
import cv2

from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
INPUT_PATH = "../input/chest-xray-pneumonia/chest_xray"
print(os.listdir(INPUT_PATH))
# list of all the training images
train_normal = Path(INPUT_PATH + '/train/NORMAL').glob('*.jpeg')
train_pneumonia = Path(INPUT_PATH + '/train/PNEUMONIA').glob('*.jpeg')

# ---------------------------------------------------------------
# Train data format in (img_path, label) 
# Labels for [ the normal cases = 0 ] & [the pneumonia cases = 1]
# ---------------------------------------------------------------
normal_data = [(image, 0) for image in train_normal]
pneumonia_data = [(image, 1) for image in train_pneumonia]

train_data = normal_data + pneumonia_data

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'])

# Checking the dataframe...
train_data.head()
train_data = train_data.sample(frac=1., random_state=100).reset_index(drop=True)

# Checking the dataframe...
train_data.head(10)
print(train_data)
# Counts for both classes
count_result = train_data['label'].value_counts()
print('Total : ', len(train_data))
print(count_result)

# Plot the results 
plt.figure(figsize=(8,5))
sns.countplot(x = 'label', data =  train_data)
plt.title('Number of classes', fontsize=16)
plt.xlabel('Class type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(range(len(count_result.index)), 
           ['Normal : 0', 'Pneumonia : 1'], 
           fontsize=14)
plt.show()
fig, ax = plt.subplots(3, 4, figsize=(20,15))
for i, axi in enumerate(ax.flat):
    image = imread(train_data.image[i])
    axi.imshow(image, cmap='bone')
    axi.set_title('Normal' if train_data.label[i] == 0 else 'Pneumonia',
                  fontsize=14)
    axi.set(xticks=[], yticks=[])
train_data.to_numpy().shape
def data_input(dataset):
    # print(dataset.shape)
    for image in dataset:
        im = cv2.imread(str(image))
        im = cv2.resize(im, (224,224))
        if im.shape[2] == 1:
            # np.dstack(): Stack arrays in sequence depth-wise 
            #              (along third axis).
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
            im = np.dstack([im, im, im])
        
        # ----------------------------------------------------------
        # cv2.cvtColor(): The function converts an input image 
        #                 from one color space to another. 
        # [Ref.1]: "cvtColor - OpenCV Documentation"
        #     - https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        # [Ref.2]: "Python计算机视觉编程- 第十章 OpenCV" 
        #     - https://yongyuan.name/pcvwithpython/chapter10.html
        # ----------------------------------------------------------
        x_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Normalization
        x_image = x_image.astype(np.float32)/255.
        return x_image
def load_data(files_dir='/train'):
    # list of the paths of all the image files
    normal = Path(INPUT_PATH + files_dir + '/NORMAL').glob('*.jpeg')
    pneumonia = Path(INPUT_PATH + files_dir + '/PNEUMONIA').glob('*.jpeg')

    # --------------------------------------------------------------
    # Data-paths' format in (img_path, label) 
    # labels : for [ Normal cases = 0 ] & [ Pneumonia cases = 1 ]
    # --------------------------------------------------------------
    normal_data = [(image, 0) for image in normal]
    pneumonia_data = [(image, 1) for image in pneumonia]
    img_data = normal_data + pneumonia_data
    # Get a pandas dataframe for the data paths 
    image_data = pd.DataFrame(img_data, columns=['image', 'label'])

    # Shuffle the data 
    image_data = image_data.sample(frac=1., random_state=100).reset_index(drop=True)

    x_images, y_labels = ([data_input(image_data.iloc[i][:]) for i in range(len(image_data))], 
                             [image_data.iloc[i][1] for i in range(len(image_data))])

    x_images = np.array(x_images)
    x_images = x_images.reshape(x_images.shape[0],x_images.shape[1]*x_images.shape[2]*x_images.shape[3])
    
    y_labels = np.array(y_labels)
    
    return x_images,y_labels
x_train, y_train = load_data(files_dir='/train')

print(x_train.shape)
print(y_train.shape)
x_test, y_test= load_data(files_dir='/test')

print(x_test.shape)
print(y_test.shape)
from sklearn.svm import SVC
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150, whiten = True, random_state=42)

svc =SVC(kernel = 'rbf',class_weight = 'balanced')
model = make_pipeline(pca, svc)
from sklearn.model_selection import GridSearchCV
param_grid ={'svc__C':[1,5,10,50],'svc__gamma':[0.0001,0.0005,0.001,0.005]}
grid = GridSearchCV(model, param_grid)

%time grid.fit(x_train,y_train)
print(grid.best_params_)
model = grid.best_estimator_
yfit = model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,yfit))