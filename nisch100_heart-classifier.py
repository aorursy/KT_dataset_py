import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns #for plotting

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dt = pd.read_csv("../input/heart.csv")
dt.head(10)
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
dt['sex'][dt['sex'] == 0] = 'female'

dt['sex'][dt['sex'] == 1] = 'male'

dt
dt['sex'] = dt['sex'].astype('object')

dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')

dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')

dt['rest_ecg'] = dt['rest_ecg'].astype('object')

dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')

dt['st_slope'] = dt['st_slope'].astype('object')

dt['thalassemia'] = dt['thalassemia'].astype('object')