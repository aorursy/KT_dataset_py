# IMPORTING NECESSARY MODULES FOR DATA ANALYSIS AND PREDICTIVE MODELLING

import numpy as np

import pandas as pd

import seaborn as sns

import psutil

import cv2

import humanize

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import xgboost as xgb

import os

import gc

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm

import matplotlib.pyplot as plt

from IPython.display import HTML, display, clear_output

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
print(os.listdir('../input/'))
#len(next(os.walk("../input/train/"))[2]) # Number of images present in the train folder
TrainDataPath = '../input/train_labels.csv'

SubDataPath = '../input/sample_submission.csv'



# Loading the Training and Test Dataset

TrainData = pd.read_csv(TrainDataPath)

SubData = pd.read_csv(SubDataPath)
print("Training Dataset Shape:")

print(TrainData.shape)

print("\n")

print("Training Dataset Columns/Features:")

print(TrainData.dtypes)

TrainData.head()
# checking missing data percentage in train data

total = TrainData.isnull().sum().sort_values(ascending = False)

percent = (TrainData.isnull().sum()/TrainData.isnull().count()*100).sort_values(ascending = False)

missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_TrainData.head(10)
SubData.head() # Submission Format
print("Number Of Training Images Present In the Train Folder:")

print(TrainData.shape[0])

print()

print("Number Of Test Images Present In the Test Folder:")

print(SubData.shape[0])

print()

print()

print("0 - Represent Presence Of NO Tumors")

print("1 - Represent Presence Of Tumors")

print()

print()

print()

print("The Evaluation Metric For This Problem :  area under the ROC curve")
def printmemusage():

 process = psutil.Process(os.getpid())

 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

printmemusage()
# source: https://www.kaggle.com/gpreda/honey-bee-subspecies-classification

def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):

    categories = (df.groupby([col_name])[col_name].nunique()).index

    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 

                         figsize=(4*figure_cols,4*len(categories))) # adjust size here

    # draw a number of images for each location

    for i, cat in enumerate(categories):

        sample = df[df[col_name]==cat].sample(figure_cols) # figure_cols is also the sample size

        for j in range(0,figure_cols):

            file=IMAGE_PATH + sample.iloc[j]['id'] + '.tif'

            im=cv2.imread(file)

            ax[i, j].imshow(im, resample=True, cmap='gray')

            ax[i, j].set_title('Label: '+str(cat), fontsize=16)  

    plt.tight_layout()

    plt.show()
def train_test_data_check(train_df, test_df, cols=None, use_all_cols=True):

    if cols == None:

        if use_all_cols:

            train_cols = set(train_df.columns)

            test_cols = set(test_df.columns)

            cols = train_cols.intersection(test_cols)

        else:

            train_cols = set(train_df.select_dtypes(['object', 'category']).columns)

            test_cols = set(test_df.select_dtypes(['object', 'category']).columns)

            cols = train_cols.intersection(test_cols)

        

    for i, col in enumerate(cols):

        display(HTML('<h3><font id="'+ col + '-ttdc' + '" color="blue">' + str(i+1) + ') ' + col + '</font></h3>'))

        print("Datatype : " + str(train_df[col].dtype) )

        print(str(train_df[col].dropna().nunique()) + " unique " + col  + " in Train dataset")

        print(str(test_df[col].dropna().nunique()) + " unique " + col  + " in Test dataset")

        extra = len(set(test_df[col].dropna().unique()) - set(train_df[col].dropna().unique()))

        print(str(extra) + " extra " + col + " in Test dataset")

        if extra == 0:

            display(HTML('<h5><font color="green"> All values present in Test dataset also present in Train dataset for column ' + col + '</font></h5>'))

        else:

            display(HTML('<h5><font color="green">' + str(extra) + ' ' +  col + ' are not present in Train dataset which are in Test dataset</font></h5>'))
def plot_bar_counts_categorical(data_se, title, figsize, sort_by_counts=False):

    info = data_se.value_counts()

    info_norm = data_se.value_counts(normalize=True)

    categories = info.index.values

    counts = info.values

    counts_norm = info_norm.values

    fig, ax = plt.subplots(figsize=figsize)

    if data_se.dtype in ['object']:

        if sort_by_counts == False:

            inds = categories.argsort()

            counts = counts[inds]

            counts_norm = counts_norm[inds]

            categories = categories[inds]

        ax = sns.barplot(counts, categories, orient = "h", ax=ax)

        ax.set(xlabel="count", ylabel=data_se.name)

        ax.set_title("Distribution of " + title)

        for n, da in enumerate(counts):

            ax.text(da, n, str(da)+ ",  " + str(round(counts_norm[n]*100,2)) + " %", fontsize=10, va='center')

    else:

        inds = categories.argsort()

        counts_sorted = counts[inds]

        counts_norm_sorted = counts_norm[inds]

        ax = sns.barplot(categories, counts, orient = "v", ax=ax)

        ax.set(xlabel=data_se.name, ylabel='count')

        ax.set_title("Distribution of " + title)

        for n, da in enumerate(counts_sorted):

            ax.text(n, da, str(da)+ ",  " + str(round(counts_norm_sorted[n]*100,2)) + " %", fontsize=10, ha='center')
train_test_data_check(TrainData, SubData)
plot_bar_counts_categorical(TrainData['label'], 'Train Dataset Column: label', (18,3))
# OpenCV uses BGR as its default colour order for images, matplotlib uses RGB.

img = cv2.imread('../input/train/'+TrainData['id'][0]+'.tif')

print(img.shape)

plt.imshow(img)
# Dsiplaying 5 images per class

IMAGE_PATH = '../input/train/' 

draw_category_images('label',5, TrainData, IMAGE_PATH)