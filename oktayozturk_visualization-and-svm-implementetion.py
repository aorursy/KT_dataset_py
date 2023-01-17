# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import dataset
data = pd.read_csv("../input/voice.csv")
# Showing five columns
data.head()
# Showing five column
data.tail()
# Describing data show us statics features
data.describe()
data.label = [1 if each == "female" else 0 for each in data.label ]
y = data.label.values
x_data = data.drop(["label"], axis = 1)
# Normalization
x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.set(style="white")
df = x.loc[:,['meandom','mindom','maxdom']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
# Plotting
data.plot(kind='scatter', x='meanfreq', y='dfrange')
data.plot(kind='kde', y='meanfreq')
# Pairplotting
sns.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], 
                 hue='label', size=3)
# Train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# Importing SVM from sklearn
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)
# Testing
print("Print accuracy of svm algorithm: ", svm.score(x_test,y_test))