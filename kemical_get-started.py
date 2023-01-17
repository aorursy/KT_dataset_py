#import commons data libs

%matplotlib inline

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns
#load data

df = pd.read_csv('../input/ks-projects-201801.csv')

df.columns
#check headers and first rows

df.head()
#check row

df.loc[df['ID'] == 1000810416]