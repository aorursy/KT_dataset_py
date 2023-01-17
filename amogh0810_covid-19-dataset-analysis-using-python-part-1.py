# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

from matplotlib import pyplot as plt
from IPython.display import display
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump, load

%matplotlib inline
abc = pd.read_csv('../input/coviddataset/COVID-19.csv')
abc.describe()
abc.head(10)
print(abc)
abc.shape
abc.info()
abc.columns
scatter_matrix(abc.loc[:,'cases':'deaths'],figsize=(12, 10))
abc.plot.scatter(x='deaths', y='cases', title='Covid-19 Dataset')
abc['deaths'].plot.hist()
abc.groupby("countryterritoryCode").deaths.mean().sort_values(ascending=False)[:5].plot.bar()
sns.heatmap(abc.corr(), annot=True)
sns.pairplot(abc)