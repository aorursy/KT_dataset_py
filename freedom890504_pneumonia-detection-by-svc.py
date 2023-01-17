import os
import glob

import numpy as np #線性代數
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
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
INPUT_PATH = "../input/pneumonia-detection/chest_xray"
print(os.listdir(INPUT_PATH))
# Training Data
train_normal = Path(INPUT_PATH + '/train/NORMAL').glob('*.jpeg')
train_pneumonia = Path(INPUT_PATH + '/train/PNEUMONIA').glob('*.jpeg')

normal_data = [(image, 0) for image in train_normal]
pneumonia_data = [(image, 1) for image in train_pneumonia]
train_data = normal_data + pneumonia_data

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'])

# Checking the dataframe...
train_data.head()
# Shuffle the data
train_data = train_data.sample(frac=1., random_state=100).reset_index(drop=True)
# Checking the dataframe...
train_data.head(10)
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
    axi.set_title('Normal' if train_data.label[i] == 0 else 'Pneumonia',fontsize=14)
    axi.set(xticks=[], yticks=[])
im = Image.open(INPUT_PATH + '/train/NORMAL/IM-0115-0001.jpeg')
nor = np.array(im)
nor.resize(224,224)
print(nor.shape)
def data_input(dataset):
    for image in dataset:
        im = cv2.imread(str(image))
        im = cv2.resize(im, (224,224))
        if im.shape[2] == 1:
            im = np.dstack([im, im, im])
        x_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
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
x_test, y_test = load_data(files_dir='/test')
print(x_test.shape)
print(y_test.shape)
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca,svc)
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C':[1, 5, 10, 50], 'svc__gamma':[0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
%time grid.fit(x_train, y_train)
print(grid.best_params_)
model =grid.best_estimator_
yfit=model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,yfit))
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True label')
