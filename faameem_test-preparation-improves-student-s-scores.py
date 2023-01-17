# to display plots within the notebook

%matplotlib inline



# import libraries

import pandas as pd

import matplotlib as mpl

import numpy as np

import seaborn as sns

import os



from matplotlib import pyplot as plt



# display library versions

print('numpy:{0}'.format(np.__version__))

print('pandas:{0}'.format(pd.__version__))

print('matplotlib:{0}'.format(mpl.__version__))

print('seaborn:{0}'.format(sns.__version__))



# misc. configurations

pd.options.display.float_format = '{:.2f}'.format

#plt.style.use('ggplot')
# check for data file

print('data file: {0}'.format(os.listdir("../input")))
# load data

data = pd.read_csv('../input/StudentsPerformance.csv')
# first 5 rows of data

data.head()
# last 5 rows of data

data.tail()
data.info()
# null data per column

data.isnull().any()
# overall null data

data.isnull().any().any()
# Determine need for standardizing/normalizing of data

data.describe().T
# categorical data - uniqe values

for column in data:

    if data[column].dtype == 'O':

        print('column: {0}\nunique values: {1}\n'.format(column, data[column].unique()))
# unique values

for column in data:

    if data[column].dtype != 'O':

        print('column: {0}\nunique values: {1}\n'.format(column, data[column].unique()))
# correlation

data.corr(method='pearson')
data['literacy score'] = (data['reading score']+data['writing score'])/2
data.head()
data2 = data.drop(['reading score', 'writing score'], axis=1)
data2.head()
def univariatePlot(column):

    data2.groupby(column).mean().plot(kind='bar', rot=45, figsize=(12,8))
def bivariatePlot(columnList):

    data2.groupby(columnList).mean().plot(kind='bar', rot=45, figsize=(12,8))
univariatePlot('gender')
bivariatePlot(['gender','test preparation course'])
univariatePlot('race/ethnicity')
bivariatePlot(['race/ethnicity','test preparation course'])
univariatePlot('parental level of education')
bivariatePlot(['parental level of education','test preparation course'])
univariatePlot('lunch')
bivariatePlot(['lunch','test preparation course'])
univariatePlot('test preparation course')