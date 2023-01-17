# IMPORTING NECESSARY MODULES FOR DATA ANALYSIS AND PREDICTIVE MODELLING

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import xgboost as xgb

import lightgbm as lgb

import re

import gc

import os

import psutil

import humanize

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm

import matplotlib.pyplot as plt

from IPython.display import HTML, display, clear_output

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
print(os.listdir("../input"))
DIR = '../input/test/'

print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
TrainDataPath = '../input/train.csv'

TestDataPath = '../input/test/seg_00030f.csv' # Randomly taking a sample test data

SubDataPath = '../input/sample_submission.csv'



# Loading the Training Dataset and Submission File

TrainData = pd.read_csv(TrainDataPath, nrows=10000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

TestData = pd.read_csv(TestDataPath, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

SubData = pd.read_csv(SubDataPath)
print("Training Dataset Shape:")

print(TrainData.shape)

print("\n")

print("Training Dataset Columns/Features:")

print(TrainData.dtypes)

TrainData.head()
print("Test Dataset Shape:")

print(TestData.shape)

print("\n")

print("Test Dataset Columns/Features:")

print(TestData.dtypes)

TestData.head()
print("Submission Dataset Shape:")

print(SubData.shape)

print("\n")

print("Submission Dataset Columns/Features:")

print(SubData.dtypes)

SubData.head()
# checking missing data percentage in train data

total = TrainData.isnull().sum().sort_values(ascending = False)

percent = (TrainData.isnull().sum()/TrainData.isnull().count()*100).sort_values(ascending = False)

missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_TrainData.head(10)
# checking missing data percentage in test data

total = TestData.isnull().sum().sort_values(ascending = False)

percent = (TestData.isnull().sum()/TestData.isnull().count()*100).sort_values(ascending = False)

missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_TrainData.head(10)
def printmemusage():

 process = psutil.Process(os.getpid())

 print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

printmemusage()
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
def count_plot_by_hue(data_se, hue_se, title, figsize, sort_by_counts=False):

    if sort_by_counts == False:

        order = data_se.unique()

        order.sort()

    else:

        order = data_se.value_counts().index.values

    off_hue = hue_se.nunique()

    off = len(order)

    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.countplot(y=data_se, hue=hue_se, order=order, ax=ax)

    ax.set_title(title)

    patches = ax.patches

    for i, p in enumerate(ax.patches):

        x=p.get_bbox().get_points()[1,0]

        y=p.get_bbox().get_points()[:,1]

        total = x

        p = i

        q = i

        while(q < (off_hue*off)):

            p = p - off

            if p >= 0:

                total = total + (patches[p].get_bbox().get_points()[1,0] if not np.isnan(patches[p].get_bbox().get_points()[1,0]) else 0)

            else:

                q = q + off

                if q < (off*off_hue):

                    total = total + (patches[q].get_bbox().get_points()[1,0] if not np.isnan(patches[q].get_bbox().get_points()[1,0]) else 0)

       

        perc = str(round(100*(x/total), 2)) + " %"

        

        if not np.isnan(x):

            ax.text(x, y.mean(), str(int(x)) + ",  " + perc, va='center')

    plt.show()
def show_unique(data_se):

    display(HTML('<h5><font color="green"> Shape Of Dataset Is: ' + str(data_se.shape) + '</font></h5>'))

    for i in data_se.columns:

        if data_se[i].nunique() == data_se.shape[0]:

            display(HTML('<font color="red"> ATTENTION!!! ' + str(i+' --> '+str(data_se[i].nunique())) + '</font>'))

        elif (data_se[i].nunique() == 1):

            display(HTML('<font color="Blue"> ATTENTION!!! ' + str(i+' --> '+str(data_se[i].nunique())) + '</font>'))

        else:

            print(i+' -->', data_se[i].nunique())
def show_countplot(data_se):

    display(HTML('<h2><font color="blue"> Dataset CountPlot Visualization: </font></h2>'))

    for i in data_se.columns:

        if (data_se[i].nunique() <= 10):

            plot_bar_counts_categorical(data_se[i].astype(str), 'Dataset Column: '+ i, (15,7))

        elif (data_se[i].nunique() > 10 and data_se[i].nunique() <= 20):

            plot_bar_counts_categorical(data_se[i].astype(str), 'Dataset Column: '+ i, (15,12))

        else:

            print('Columns do not fit in display '+i+' -->', data_se[i].nunique())
gc.collect() # Python garbage collection module for dereferencing the memory pointers and making memory available for better usage
TrainData.head()
TestData.head()
# show_unique function shows the no of unique values present in the each column of the dataset

show_unique(TrainData)
show_unique(TestData)
# TRAIN DATA HeatMap

f,ax = plt.subplots(figsize=(10, 5))

sns.heatmap(TrainData.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax,cmap='Blues')
plt.figure(figsize=(18, 5))

sns.distplot((TrainData["acoustic_data"]))

plt.title('TRAIN DATA')

plt.show()



plt.figure(figsize=(18, 5))

sns.distplot((TrainData["time_to_failure"]))

plt.title('TRAIN DATA')

plt.show()
fig, ax = plt.subplots(2,1, figsize=(18,12))

ax[1].plot(TrainData.index.values, TrainData.time_to_failure.values, c="darkred")

ax[1].set_title("Quaketime of 10 Mio rows")

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Quaketime in ms");

ax[0].plot(TrainData.index.values, TrainData.acoustic_data.values, c="mediumseagreen")

ax[0].set_title("Signal of 10 Mio rows")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("Acoustic Signal");
fig, ax = plt.subplots(3,1,figsize=(18,18))

ax[0].plot(TrainData.index.values[0:50000], TrainData.time_to_failure.values[0:50000], c="Red")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("Time to quake")

ax[0].set_title("How does the second quaketime pattern look like?")

ax[1].plot(TrainData.index.values[0:49999], np.diff(TrainData.time_to_failure.values[0:50000]))

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Difference between quaketimes")

ax[1].set_title("Are the jumps always the same?")

ax[2].plot(TrainData.index.values[0:4000], TrainData.time_to_failure.values[0:4000])

ax[2].set_xlabel("Index from 0 to 4000")

ax[2].set_ylabel("Quaketime")

ax[2].set_title("How does the quaketime changes within the first block?");