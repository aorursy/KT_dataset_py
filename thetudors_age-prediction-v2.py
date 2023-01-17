# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#print(check_output(["ls", "../input/regression_sample"]).decode("utf8"))

import shutil


import scipy.stats

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from keras import applications, optimizers, Input
from keras.models import Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv
%matplotlib inline

DIRCSV = '../input/ageprecsv1/train_gt/train_gt/train_gt.csv'
DIR = '../input/agepre7/train/train'
IMG_SIZE=150

dict_age = {'(0, 2)' : 0,
                '(4, 6)' : 1,
                '(8, 12)' : 2,
                '(15, 20)' : 3,
                '(25, 32)' : 4,
                '(38, 43)' : 5,
                '(48, 53)' : 6,
                '(60, 100)' : 7}

def label_img(name):
    #word_label = name.split('.')[1]
    word_label = name
    
    
    if int(word_label) >= 0 and int(word_label) <= 2 : return "(0,2)"
    elif int(word_label) >= 3 and int(word_label) <= 6 : return "(3,6)"
    elif int(word_label) >= 7 and int(word_label) <= 14 : return "(7,14)"
    elif int(word_label) >= 15 and int(word_label) <= 23 : return "(15,23)"
    elif int(word_label) >= 24 and int(word_label) <= 32 : return "(24,32)"
    elif int(word_label) >= 33 and int(word_label) <= 43 : return "(33,43)"
    elif int(word_label) >= 44 and int(word_label) <= 53 : return "(44,53)"
    elif int(word_label) >= 54 and int(word_label) <= 65 : return "(54,65)"
    elif int(word_label) >= 66 and int(word_label) <= 100 : return "(66,100)"
    else : print("Dosya ismi hatalı veya dosya formatı hatalı.") 
image_list = []
label_list = []
def make_train_data(DIRCSV):
    imagecount = 0
    with open(DIRCSV) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
    
        for row in readCSV:
            
            if(imagecount > 500):
                return
        
            imagecount = imagecount + 1                         
            ad = row[0]
            yas = row[1].split('.')[0]
            label = label_img(yas)
            path = os.path.join(DIR,ad)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            #print("Path + ",path)
        
            image_list.append(np.array(img))
            label_list.append(label)
make_train_data(DIRCSV)
images = np.array(image_list)
labels = np.array(label_list)
print(images.shape, labels.shape)
def extract_coins(img, to_size=100):
    """
    Find coins on the image and return array
    with all coins in (to_size, to_size) frame 
    
    return (n, to_size, to_size, 3) array
           array of radiuses fo coins
    n - number of coins
    color map: BGR
    """
    # Convert to b&w
    cimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Find circles on the image
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 2, 60, param1=300, param2=30, minRadius=30, maxRadius=50)
    
    # Convert to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define color range for masking
    lower = np.array([0,0,0])
    upper = np.array([255,255,90])
    # Apply the mask
    mask = cv2.blur(cv2.inRange(hsv, lower, upper), (8, 8))
    
    
    frames = []
    radiuses = []
    # If circles were not found
    if circles is None:
        return None, None
    
    for circle in circles[0]:
        
        center_x = int(circle[0])
        center_y = int(circle[1])
        
        # If center of coin lays in masked coin range
        if not mask[center_y, center_x]:
            continue
        
        # increase radius by C
        # circle detector tends to decrease radius
        radius = circle[2] + 3
        
        radiuses.append(radius)
        
        # Coordinates of upper left corner of square
        x = int(center_x - radius)
        y = int(center_y - radius)
        
        # As radius was increased the coordinates
        # could go out of bounds
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        
        # Scale coins to the same size
        resized = cv2.resize(img[y: int(y + 2 * radius), x: int(x + 2 * radius)], 
                             (to_size, to_size), 
                             interpolation = cv2.INTER_CUBIC)

        frames.append(resized)

    return np.array(frames), radiuses
scaled = []
scaled_labels = []
radiuses = []
for nominal, image in zip(labels, images):
    prepared, radius = extract_coins(image)
    if prepared is not None and len(prepared):
        scaled.append(prepared[0])
        scaled_labels.append(nominal)
        radiuses.append(radius[0])

# Create dataframe with data and pickle it
data = pd.DataFrame({'label': scaled_labels, 'radius': radiuses, 'image': scaled})
data.to_pickle('file.pickle')
# Load data
data = pd.read_pickle('file.pickle')
# Radiuses distribution
data.groupby('label').mean().plot.bar()
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.radius.values, data.label.values, test_size=0.20, random_state=42)
X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)
clf = SVC()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)