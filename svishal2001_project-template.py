
# This notebook is a template to give directions on our expectations and serve as a proxy for a rubric as well. 
# You will find a copy of this in the Vocareum workspace as well. If you do decide to use notebooks on kaggle, 
# do keep your work private, or just share with your team mates only. 
# Packages and libraries load here [basic packages are specified; additional packages may be needed]
%matplotlib inline

from pathlib import Path

import pandas as pd
import numpy as np

import missingno as msno
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

import matplotlib.pylab as plt

#from dmba import regressionSummary, adjusted_r2_score, AIC_score, BIC_score
# Load Data set here
# Input data files are available in the "../input/" directory.
# For example, running this cell(by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Initial inspection begins here
# Data Visualization begins here
# Visualize pairwise correlations and comment here
#Check for missing values here and find ways to handle them.
# Inspect categorical variables here
# Establish X (predictors) and y (response)
#Encode Categorical variables
# Split the data here.  
# data processing here 
# Model building here 
# Performance Analysis 