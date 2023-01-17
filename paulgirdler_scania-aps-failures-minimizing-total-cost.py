# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib as plt #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(plt.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

# misc libraries
import random
import time


# ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

# Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
# Set style parameters
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
# All NaN are listed as "na"

#import data from file: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
data_raw = pd.read_csv('../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv', na_values = "na")

#a dataset should be broken into 3 splits: train, test, and (final) validation
#the test file provided is the validation file for competition submission
#we will split the train set into train and test data in future sections
data_val  = pd.read_csv('../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv', na_values = "na")

#to play with our data we'll create a copy
#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs
data1 = data_raw.copy(deep = True)

#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]

#preview data
print (data_raw.info()) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html
#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html
data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
print("No. of columns not containing null values")
print(len(data1.columns[data1.notna().all()]))
print("-"*10)

print("Total no. of columns in the dataframe")
print(len(data1.columns))
print("-"*10)

data1.describe(include = 'all')
missing = data1.isna().sum().div(data1.shape[0]).mul(100).to_frame().sort_values(by=0, ascending = False)
fig = missing.plot(kind='bar', figsize=(50,20), fontsize=24)
fig.set_title("Plot:  Sorted Proportion of Missing Values by Column for Training Set", fontsize=36)
fig.set_xlabel("Columns", fontsize=36)
fig.set_ylabel("Proportion Missing (%)", fontsize=36)
###COMPLETING: complete or delete missing values in train and test/validation dataset

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

# Select columns from the training set with more than 30% missing values
drop_column = data1.columns[data1.isnull().mean() > 0.3]

# Process both datasets
for dataset in data_cleaner:
    dataset.rename(columns={"class":"Class"}, inplace=True) #to avoid name collision with the class
    # Encode class labels
    dataset['Class'] = lb.fit_transform(dataset['Class'])
    
    # Delete selected features with too many missing values from both training sets
    dataset.drop(drop_column, axis=1, inplace = True)
    
    for col in dataset:
        # Complete missing col with mode
        dataset[col].fillna(dataset[col].mean(), inplace = True)
print("No. of columns not containing null values")
print(len(data1.columns[data1.notna().all()]))
print("-"*10)

print("Total no. of columns in the dataframe")
print(len(data1.columns))
print("-"*10)

data1.describe(include = 'all')
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

# define y variable aka target/outcome
Target = ['Class']

# define x variables for original features aka feature selection
data1_x = data1.columns.drop('Class')
# split train and test data with function defaults
# random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x], data1[Target], random_state = 0)

print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x.head()
# Discrete Variable Correlation by Failure using
# group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Failure Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
# https://www.kaggle.com/percevalve/scania-dataset-eda-for-histograms

from collections import Counter

def get_tag(name):
    return name.split("_")[0]

columns_list = train1_x.columns

all_columns_with_tags = [a for a in columns_list if "_" in a]
all_tags = [get_tag(a) for a in all_columns_with_tags]
hists = [k for k, v in Counter(all_tags).items() if v == 10]
hists_columns = [k for k in all_columns_with_tags if get_tag(k) in hists]
hists_dict = {k:[col for col in hists_columns if k in col] for k in hists if get_tag(k) in hists}
counter_columns = [k for k in all_columns_with_tags if get_tag(k) not in hists]
# https://www.kaggle.com/percevalve/scania-dataset-eda-for-histograms
    
for hist in hists:
    data1[f"{hist}_total"] = sum(data1[col] for col in hists_dict[hist])
data1["system_age"] = data1[[f"{hist}_total" for hist in hists]].max(axis=1)

plt.figure(figsize=(15,5));
for_plotting = data1[data1.system_age>=0]
_,bins,_ = plt.hist(np.log(for_plotting[for_plotting.Class==0].system_age+1),bins=100,density=True,alpha=0.5,label="Class 0");
plt.hist(np.log(for_plotting[for_plotting.Class==1].system_age+1),bins=bins,density=True,alpha=0.5,label="Class 1");
plt.legend();
plt.ylabel("Percentage per categorie");
plt.xlabel("Number of measurements (a.k.a. System Age) for physical (log scale)");
plt.xlim(0,21);
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix

def my_scorer(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = 10*fp+500*fn
    return cost

my_func = make_scorer(my_scorer, greater_is_better=False)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint

# Number of components for pca
n_components = [int(x) for x in np.linspace(start = 5, stop = 26, num = 1)]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 50]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]

# Method of selecting samples for training each tree
bootstrap = [True, False]

clf = RandomForestClassifier(class_weight="balanced", random_state=0)
pca = PCA()

pipe = Pipeline(steps=[("pca", pca),("clf", clf)])

# Create the random grid
random_grid = {'pca__n_components': n_components,
               'clf__n_estimators': n_estimators,
               'clf__max_features': max_features,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
               'clf__bootstrap': bootstrap
              }

# Look at parameters used by random forest
print('Parameters currently in use:\n')
pprint(clf.get_params())

search = RandomizedSearchCV(pipe, random_grid, iid = False, cv = 5, return_train_score = True, scoring = my_func, n_jobs = -1, verbose=3)
search.fit(train1_x, train1_y)

# Printing best classificator
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
# Printing best classificator on test set
test_score = - search.score(test1_x, test1_y)
test_score_per_truck = test_score/test1_x.shape[0]

print("Best model on test set (Cost = $ %0.2f):" % test_score)
print("Best model cost per truck on test set (Cost = $ %0.2f)" % test_score_per_truck)