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
# Regular Exploratory data analysis and plotting libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



#Models to import from scikit-learn

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



# Model Evaluations

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import plot_roc_curve
heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart.shape
heart.head(20)
heart.tail(20)
heart["target"].value_counts()
heart["target"].value_counts().plot(kind="bar", color=["yellow","lightblue"]);
#check missing values

heart.isna()
#check info about the dataset

heart.info()
heart.describe()