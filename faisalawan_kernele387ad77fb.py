import os

import sys

from pandas import Series

import pandas as pd

import numpy as np

import traceback

import time

import seaborn as sns

import matplotlib.pyplot as plt

import chardet
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
with open('../input/libraryevents/library_events.csv', 'rb') as f:

    result = chardet.detect(f.read())
df = pd.DataFrame(columns=['Title', 'Description','Date','StartTime','EndTime','Location','Library','Categories'])

iteri=0
print(os.listdir("../input"))
data = pd.read_csv("../input/libraryevents/library_events.csv", encoding=result['encoding'])

data.head()
print('Shape of the data set: ' + str(data.shape))
print("describe: ")

print(data.describe())
print(data.info())
label = ['Main Library','George Reynolds Branch','Meadows Branch','NoBo Corner Library','*Off-Site','Carnegie Library for Local History','All Locations']





no_lib = [

    9157,

    1830,

    1877,

    876,

    68,

    10,

    2

]




index = np.arange(len(label))

plt.bar(index, no_lib)

plt.xlabel('Locations', fontsize=5)

plt.ylabel('No of Library Events', fontsize=5)

plt.xticks(index, label, fontsize=5, rotation=30)

plt.title('Events in the Library')

plt.show()



cate = ['Classes & Activities','Storytime','STEAM','Discussion Groups','Cinema','Concerts','Performances & Presentations','Exhibitions','Workshop','Tool Orientation','Open Studio', 'Guided Practice','Summer of Discovery','Family Event','BoulderReads']



x = ['Author Talks','Outreach','Sponsored Library Program','Boulder Office of Arts + Culture','Literacy','Storytime','Social Connections','Pollinator Applications','Civic Life']
no_cat = [

    1461,

    1716,

    143,

    463,

    117,

    41,

    70,

    13,

    57,

    12,

    18,

    10,

    57,

    6,

   101,

    

]
indexx = np.arange(len(cate))

plt.bar(indexx, no_cat)

plt.xlabel('Categories', fontsize=5)

plt.ylabel('No of Events', fontsize=5)

plt.xticks(indexx, cate, fontsize=5, rotation=30)

plt.title('Categories Events in the Library')

plt.show()