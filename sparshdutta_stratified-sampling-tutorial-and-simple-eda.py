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
DataPath = '../input/heart.csv'



# Loading the Training Dataset and Submission File

Data = pd.read_csv(DataPath)
print("Heart Dataset Shape:")

print(Data.shape)

print("\n")

print("Heart Dataset Columns/Features:")

print(Data.dtypes)

Data.head()
# checking missing data percentage in heart dataset

total = Data.isnull().sum().sort_values(ascending = False)

percent = (Data.isnull().sum()/Data.isnull().count()*100).sort_values(ascending = False)

missing_Data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_Data.head(20)
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

            print('{:10s} --->   {} unique values'.format(i, data_se[i].nunique())) 
def show_countplot(data_se):

    display(HTML('<h2><font color="blue"> Dataset CountPlot Visualization: </font></h2>'))

    for i in data_se.columns:

        if (data_se[i].nunique() <= 10):

            plot_bar_counts_categorical(data_se[i], 'Dataset Column: '+ i, (18,5))

        elif (data_se[i].nunique() > 10 and data_se[i].nunique() <= 20):

            plot_bar_counts_categorical(data_se[i], 'Dataset Column: '+ i, (18,12))

        else:

            print('Columns do not fit in display {:10s} --->   {} unique values'.format(i, data_se[i].nunique())) 
gc.collect() # Python garbage collection module for dereferencing the memory pointers and making memory available for better usage
Data.head()
Data.info()
Data.describe().T
Data.dtypes
# show_unique function shows the no of unique values present in each column of the dataset

show_unique(Data)
# Heart Data HeatMap

f,ax = plt.subplots(figsize=(18, 10))

sns.heatmap(Data.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax,cmap='Greens_r')
plot_bar_counts_categorical(Data['target'], "Heart Data Column: target", figsize=(18,5), sort_by_counts=False)
show_countplot(Data)
Data.head()
plt.figure(figsize=(18, 5))

sns.distplot(Data['age'])

plt.title('Heart Data Column : "age" Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(Data['age'])

plt.title('Heart Data Column : "age" Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(Data['trestbps'])

plt.title('Heart Data Column : "trestbps" Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(Data['trestbps'])

plt.title('Heart Data Column : "trestbps" Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(Data['chol'])

plt.title('Heart Data Column : "chol" Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(Data['chol'])

plt.title('Heart Data Column : "chol" Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(Data['thalach'])

plt.title('Heart Data Column : "thalach" Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(Data['thalach'])

plt.title('Heart Data Column : "thalach" Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(Data['oldpeak'])

plt.title('Heart Data Column : "oldpeak" Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(Data['oldpeak'])

plt.title('Heart Data Column : "oldpeak" Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(np.log1p(Data['trestbps']))

plt.title('Heart Data Column : "trestbps" Log1p Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(np.log1p((Data['trestbps'])))

plt.title('Heart Data Column : "trestbps" Log1p Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(np.log1p(Data['chol']))

plt.title('Heart Data Column : "chol" Log1p Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(np.log1p((Data['chol'])))

plt.title('Heart Data Column : "chol" Log1p Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(np.log1p(Data['thalach']))

plt.title('Heart Data Column : "thalach" Log1p Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(np.log1p((Data['thalach'])))

plt.title('Heart Data Column : "thalach" Log1p Box Plot')

plt.show()
plt.figure(figsize=(18, 5))

sns.distplot(np.log1p(Data['oldpeak']))

plt.title('Heart Data Column : "oldpeak" Log1p Distribution Plot')

plt.show()



plt.figure(figsize=(18, 5))

sns.boxplot(np.log1p((Data['oldpeak'])))

plt.title('Heart Data Column : "oldpeak" Log1p Box Plot')

plt.show()
Data.head()
sns.jointplot("age", "trestbps", data=Data, kind="reg")

sns.jointplot("age", "trestbps", data=Data, kind="hex")

sns.jointplot("age", "trestbps", data=Data, kind="kde")

sns.jointplot("age", "trestbps", data=Data, kind="scatter")
count_plot_by_hue(Data['thal'], Data['target'], "Heart Data Column : 'thal' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['ca'], Data['target'], "Heart Data Column : 'ca' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['slope'], Data['target'], "Heart Data Column : 'slope' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['exang'], Data['target'], "Heart Data Column : 'exang' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['restecg'], Data['target'], "Heart Data Column : 'restecg' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['fbs'], Data['target'], "Heart Data Column : 'fbs' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['cp'], Data['target'], "Heart Data Column : 'cp' vs 'target'", figsize=(18,5), sort_by_counts=False)
count_plot_by_hue(Data['sex'], Data['target'], "Heart Data Column : 'sex' vs 'target'", figsize=(18,5), sort_by_counts=False)
Datframe = pd.DataFrame([])

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state = 5) # n_splits denotes in how many parts we need to break the original dataset

target = Data['target']



idx = target.index.values

np.random.shuffle(idx)



for trn_idx, _ in folds.split(target, target):

    Datframe = pd.DataFrame([])

    Datframe['target'] = target.loc[idx[trn_idx]].values

    plot_bar_counts_categorical(Datframe['target'], 'Heart Dataset Column: "target"', (18,3))