!pip install pygam
!pip install graphviz
!pip install pydotplus
import numpy as np 

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as sam

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import PolynomialFeatures

from pygam import LogisticGAM,LinearGAM, s, f

from pygam.datasets import default, wage

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus   

from patsy import dmatrix

import statsmodels.formula.api as smf

from sklearn import linear_model as lm

from matplotlib import pyplot



import os

print(os.listdir("../input/datalab6"))