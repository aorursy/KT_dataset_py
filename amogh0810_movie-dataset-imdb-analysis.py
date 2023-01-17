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

from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import lag_plot
abc= pd.read_csv('../input/movie-dataset-imdb/movie_dataset.csv')
abc.head(5)
abc.columns
abc.plot.scatter(x='budget', y='revenue', title='IDBM Movie Dataset')
abc.groupby("genres").popularity.mean().sort_values(ascending=False)[:5].plot.bar()
abc.groupby("original_title").popularity.mean().sort_values(ascending=False)[:5].plot.bar()
abc.groupby("original_title").budget.mean().sort_values(ascending=False)[:5].plot.bar()
abc.groupby("original_title").revenue.mean().sort_values(ascending=False)[:5].plot.bar()
abc.groupby("original_title").vote_count.mean().sort_values(ascending=False)[:5].plot.bar()
abc.plot.scatter(x='vote_count', y='vote_average', title='IDBM Movie Dataset')
sns.heatmap(abc.corr(), annot=True)
sns.pairplot(abc)
series = read_csv('../input/movie-dataset-imdb/movie_dataset.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

series.plot()

pyplot.show()
