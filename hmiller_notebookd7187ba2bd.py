import pandas as pd

from pandas import Series, DataFrame

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

sns.set_style('whitegrid')

%matplotlib inline

from sklearn.linear_model import LogisticRegression 

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

titanic_df= pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df   = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)

titanic_df.head()



titanic_df.info()

print("----------------")

test_df.info()