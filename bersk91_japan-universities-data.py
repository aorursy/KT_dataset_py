import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import graphviz

from sklearn.metrics import confusion_matrix,roc_curve,auc,roc_auc_score



from sklearn.linear_model import LogisticRegression

# helpful character encoding module

import chardet



import os

print(os.listdir("../input"))
with open("../input/school_search_20180330_1.csv", 'rb') as rawdata:

    result = chardet.detect(rawdata.read(10000))



# check what the character encoding might be

print(result)

            

#data = pd.read_csv('../input/school_search_20180330_1.csv')



#data.head()
# read in the file with the encoding detected by chardet

data = pd.read_csv("../input/school_search_20180330_1.csv", encoding='ISO-8859-1')



# look at the first few lines

data.head()