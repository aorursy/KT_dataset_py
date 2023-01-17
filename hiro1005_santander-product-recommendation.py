# Data file
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
train = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv.zip', header=0)
train.head(10)
test = pd.read_csv('/kaggle/input/santander-product-recommendation/test_ver2.csv.zip', header=0)
test.head(10)
submission = pd.read_csv('/kaggle/input/santander-product-recommendation/sample_submission.csv.zip', header=0)
submission.head(10)
