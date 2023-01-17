%matplotlib inline



#%load_ext autoreload  # Autoreload has a bug : when you modify function in source code and run again, python kernel hangs :(

#%autoreload 2



import datetime as dt



import sys, importlib



from functions_py import * # MODIFIED for kaggle (replaced by functions_py instead of functions)

importlib.reload(sys.modules['functions_py']) # MODIFIED for kaggle



#from display_factorial import *

#importlib.reload(sys.modules['display_factorial'])



import pandas as pd



pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)



import datetime as dt



import os

import zipfile

import urllib



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np    



import ipywidgets as widgets



import qgrid



import glob



from pandas.plotting import scatter_matrix



from sklearn.model_selection import StratifiedShuffleSplit





from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.metrics import pairwise_distances



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



from sklearn.metrics import classification_report



from sklearn.metrics import confusion_matrix



#from yellowbrick.classifier import ROCAUC

from sklearn.metrics import roc_auc_score



import codecs





DATA_PATH = os.path.join("../input", "stanford-dogs-dataset", "images") # Modified for kaggle

DATA_PATH = os.path.join(DATA_PATH, "Images")





DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")

ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)



ALL_FEATURES = []



plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib



import seaborn as sns

from seaborn import boxplot

sns.set()



#import common_functions



####### Paramètres pour sauver et restaurer les modèles :

import pickle

####### Paramètres à changer par l'utilisateur selon son besoin :





from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error





### For progress bar :

#from tqdm import tqdm_notebook as tqdm  #Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`

from tqdm.notebook import tqdm



# Statsmodel : 

import statsmodels.formula.api as smf



import statsmodels.api as sm

from scipy import stats



from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn import preprocessing
from PIL import Image

from io import BytesIO
from keras.applications.vgg16 import VGG16

from keras.layers import Dense





from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Flatten



from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input



from keras.applications.vgg16 import decode_predictions



import keras
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#os.environ["AUTOGRAPH_VERBOSITY"] = "10"

#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



from platform import python_version

import warnings

import time

import datetime as dt

from sklearn.metrics import classification_report, confusion_matrix

import multiprocessing as mp

import shutil



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.utils import *

from tensorflow.keras.callbacks import *

from tensorflow.keras.initializers import *



import pandas as pd

import numpy as np

import seaborn as sn



from PIL import Image

import xml.etree.ElementTree as ET

import psutil

import random



warnings.filterwarnings("ignore")

%matplotlib inline



print("py", python_version())

print("tf", tf.__version__)

print("keras", tf.keras.__version__)

mem = psutil.virtual_memory()

print("mem", mem.total/1024/1024)

cpu = mp.cpu_count()

print("cpu", cpu)



#%system nvidia-smi

#%system rocm-smi
# Those variables must be consisten with what first notebook has been ran with

NB_CLASSES = 120
# For cropping and loading of images: sourced from https://www.kaggle.com/devang/transfer-learning-with-keras-and-efficientnets
epochs = 100

batch_size = 100

testsplit = .2

targetx = 224

targety = 224

learning_rate = 0.0001

classes = 120

seed = random.randint(1, 1000)



data_dir = "/kaggle/input/stanford-dogs-dataset/images/Images/"

annotations_dir = "/kaggle/input/stanford-dogs-dataset/annotations/Annotation/"

cropped_dir = "/kaggle/working/cropped/"
%system rm -rf $cropped_dir

%system mkdir $cropped_dir



#this function adapted from https://www.kaggle.com/hengzheng/dog-breeds-classifier

def save_cropped_img(path, annotation, newpath):

    tree = ET.parse(annotation)

    xmin = int(tree.getroot().findall('.//xmin')[0].text)

    xmax = int(tree.getroot().findall('.//xmax')[0].text)

    ymin = int(tree.getroot().findall('.//ymin')[0].text)

    ymax = int(tree.getroot().findall('.//ymax')[0].text)

    image = Image.open(path)

    image = image.crop((xmin, ymin, xmax, ymax))

    image = image.convert('RGB')

    image.save(newpath)



def crop_images():

    breeds = os.listdir(data_dir)

    annotations = os.listdir(annotations_dir)



    print('breeds: ', len(breeds), 'annotations: ', len(annotations))



    total_images = 0



    for breed in breeds:

        dir_list = os.listdir(data_dir + breed)

        annotations_dir_list = os.listdir(annotations_dir + breed)

        img_list = [data_dir + breed + '/' + i for i in dir_list]

        os.makedirs(cropped_dir + breed)



        for file in img_list:

            annotation_path = annotations_dir + breed + '/' + os.path.basename(file[:-4])

            newpath = cropped_dir + breed + '/' + os.path.basename(file)

            save_cropped_img(file, annotation_path, newpath)

            total_images += 1

    

    print("total images cropped", total_images)



crop_images()