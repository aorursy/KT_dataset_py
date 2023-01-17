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
df = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding='ISO-8859-1')
df.head()
df.describe()
df.shape
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import datetime, nltk, warnings

import matplotlib.cm as cm

import itertools

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import preprocessing, model_selection, metrics, feature_selection

from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn import neighbors, linear_model, svm, tree, ensemble

from wordcloud import WordCloud, STOPWORDS

from sklearn.ensemble import AdaBoostClassifier

from sklearn.decomposition import PCA

from IPython.display import display, HTML

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

%matplotlib inline
#______

# read the datafile

df_initial = pd.read_csv('/kaggle/input/ecommerce-data/data.csv',encoding="ISO-8859-1",

                         dtype={'CustomerID': str,'InvoiceNo': str})

print('Dataframe dimensions de robby:', df_initial.shape)
df_initial.head()
df_initial['Description'].value_counts()
df_initial['Description'].value_counts()[:15].plot(kind='bar',figsize=(13,4))
df_initial['CustomerID'].value_counts()[:15].plot(kind='bar',figsize=(13,4))
#__

df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])

#____________________

# gives some infos on columns types and numer of null values

tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

display(tab_info)

#______

# show first lines

display(df_initial[:5])