

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from sklearn.metrics import precision_score, recall_score

from sklearn import preprocessing

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

t = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

tr = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
t.head()

tr.head()