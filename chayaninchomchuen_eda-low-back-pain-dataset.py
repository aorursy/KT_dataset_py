# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data=pd.read_csv("../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv") 

data.head()

data.shape

data.info()
import pandas as pd

from sklearn.preprocessing import StandardScaler 

data=pd.read_csv("../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv") 

data.drop('Unnamed: 13', axis=1, inplace=True) #drop unwated colums

data=pd.concat([data,pd.get_dummies(data['Class_att'])],axis=1)# convet categorical feature using dummies

data.drop(['Class_att','Normal'], axis=1, inplace=True)

data.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 

                'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 

                'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Labels']# Rename columns



num_ab=data.Labels.sum()  # number of abnormal label

num_nor= data.Labels.count()-num_ab  # number of normal label



data.describe() # statiscal analysis



import matplotlib.pyplot as plt # Visualize outlier using boxplot

plt.subplots(figsize=(15,6))



data.boxplot(patch_artist=True, sym="k.")

plt.xticks(rotation=90)

# correcting outlier using IQR score

minimum = 0

maximum = 0



def detect_outlier(feature):

    q1, q3= np.percentile(X[feature],[25,75])

    iqr = q3 - q1

    lower= q1 -(1.5 * iqr) 

    upper= q3 +(1.5 * iqr) 

    X.loc[X[feature] < lower, feature] = X[feature].median() 

    X.loc[X[feature] > upper, feature] = X[feature].median()

X = data.iloc[:, :-1]

for i in range(len(X.columns)): 

        detect_outlier(X.columns[i])

plt.subplots(figsize=(15,6))

X.boxplot(patch_artist=True, sym="k.")

plt.xticks(rotation=90)

# observe again with boxplot
import seaborn as sns

import matplotlib.pyplot as plt

plt.subplots(figsize=(12,8))

hm = sns.heatmap(data.corr(), cmap='YlGnBu',annot=True)

import seaborn as sns

import matplotlib.pyplot as plt

sns.pairplot(data, hue="Labels")
from sklearn.preprocessing import StandardScaler 

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

Scaler = StandardScaler()

pca=PCA()

pipeline=make_pipeline(Scaler,pca)

pipeline.fit(data)

features=range(pca.n_components_)

non_standard=plt.figure(figsize=(10,8))

plt.bar(features,pca.explained_variance_)

plt.xticks(features)

plt.title('Observe intrinsic dimension')

plt.ylabel('Variance')

plt.xlabel('PCA feature')




