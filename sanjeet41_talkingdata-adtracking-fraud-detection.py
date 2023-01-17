

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
color = sns.color_palette()
print(os.listdir("../input"))
import csv
# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/train_sample.csv")
print("\n",dataset.head())
print(dataset.shape)
dataset.isnull().sum().sort_values(ascending = False)

dataset.fillna(0,inplace = True)
dataset.isnull().sum().sort_values(ascending = False)
def normalizeData(dataset):
    # Instantiate Scaler Object 
    ss = StandardScaler()
    # Fit and transform 
    ss.fit_transform(dataset)
    dataset.head(100000)
    #country.head(10)
    return dataset
def univariateAnalysis(dataset):
    sns.distplot(dataset['ip'])

#heatmapAnaly(testdata)
univariateAnalysis(dataset)
dataset.groupby("os").count()
dataset.groupby("app").count()
dataset.groupby("device").count()
dataset.groupby("channel").count()

dataset['is_attributed'].fillna(0,inplace=True)
dataset['is_attributed'].value_counts()
%matplotlib inline
pd.crosstab(dataset.device,dataset.is_attributed).plot(kind='bar')
plt.title('Device with Is_Attribted')
plt.xlabel('Device')
plt.ylabel('Is_Attributed')