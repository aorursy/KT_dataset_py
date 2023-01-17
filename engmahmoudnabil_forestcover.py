# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
#from scipy.stats import linregress
import scipy
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.import zipfilepath.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import zipfile

zf=zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
zf2=zipfile.ZipFile('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')

sample_submission= pd.read_csv('../input/forest-cover-type-kernels-only/sample_submission.csv.zip')
sampleSubmission= pd.read_csv('../input/forest-cover-type-kernels-only/sampleSubmission.csv.zip')
train= pd.read_csv(zf.open('train.csv'))
test= pd.read_csv(zf2.open('test.csv'))

print('train (r,c)=',train.shape)
print('test (r,c)=',test.shape)
print('sample_submission (r,c)=',sample_submission.shape)
print('sampleSubmission (r,c)=',sampleSubmission.shape)
train.head()
test.head().describe()
test.head()
test.head().describe()
sample_submission.head()
sample_submission.head().describe()
sampleSubmission.head()
sampleSubmission.head().describe()
train.set_index('Id')
test.set_index('Id')
sample_submission.set_index('Id')
sampleSubmission.set_index('Id')
print('Number of null values in the dataset=', len(train[train.isnull()]) )
train.fillna('unknown')
plt.plot(train.Elevation, train.Aspect)
plt.show()

# plt.show()
#first we group the data by its elevation and compare it by the number of aspects , minumin of aspects and max of aspects
p=train.groupby(['Elevation']).Aspect.agg([len, min, max])

#plt.scatter(p)
plt.plot(p.len)
plt.xlabel('Elevation')
plt.ylabel('Number of Apects')
plt.show()  # or plt.savefig("name.png")
# plt.scatter(x, y)
# y = train.
# z=
# plt.scatter(train.Elevation, )
# plt.show()  # or plt.savefig("name.png")
p=train.groupby(['Elevation']).Aspect.mean()
p.astype('int')
plt.plot(p)
plt.show()
x=train.Elevation.mean()
plt.scatter(x,train.Aspect.mean())

plt.show()
#row1 is Id OF train row2 is Id of Cover_Type
# def add(row1,row2) :
#     if (row1.Id == row2.Id):
#         x= pd.concat([row1, row2.Cover_Type])
#     else:
#         print('unkown value')
#     return x    
# 
left= train.set_index('Id')
right= sample_submission.set_index('Id')
output=left.join(right, lsuffix='_')
final_data = output.drop(columns="Cover_Type")
final_data

final_data.to_csv('FinalSampleSubmition', index = False)