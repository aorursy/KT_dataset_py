# Load in our libraries

import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(10)