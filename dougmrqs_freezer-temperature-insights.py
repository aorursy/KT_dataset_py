from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os
data = pd.read_csv('../input/Geladeira.csv')
print(data.head())
print()
print(data.sample(10))
# I needed to create a Index column to orient myself about the order of value's aquisition

data['Index'] = range(1, len(data) + 1)
print(data.head())
data.shape

g = sns.FacetGrid(data, hue="TempC",)
g = g.map(plt.scatter, "HIC", "Hum",edgecolor="w");
pd.plotting.scatter_matrix(data,figsize=(7,7))
plt.figure()
f,ax=plt.subplots(1,3,figsize=(18,4))

sns.jointplot(x='TempC',y='TempF' ,data=data, kind='reg', ax=ax[0])
ax[0].set_title('TempC vs TempF')
ax[0].set_yticks(range(0,110,10))
sns.jointplot(x='HIC',y='HIF' ,data=data, kind='reg',ax=ax[1])
ax[1].set_title('HIC vs HIF')
ax[1].set_yticks(range(0,110,10))
sns.jointplot(x='HIC',y='Hum' ,data=data, kind='reg',ax=ax[2])
ax[2].set_title('HIC vs Hum')
ax[2].set_yticks(range(0,110,10))
plt.show()
f,ax=plt.subplots(1,3,figsize=(18,4))

sns.swarmplot(x='TempC',y='TempF' ,data=data, ax=ax[0])
ax[0].set_title('TempC vs TempF')
ax[0].set_yticks(range(0,110,10))
sns.swarmplot(x='HIC',y='HIF' ,data=data,ax=ax[1])
ax[1].set_title('HIC vs HIF')
ax[1].set_yticks(range(0,110,10))
sns.swarmplot(x='HIC',y='Hum' ,data=data,ax=ax[2])
ax[2].set_title('HIC vs Hum')
ax[2].set_yticks(range(0,110,10))
plt.show()
sns.lineplot(x='Index', y='TempC',data=data)