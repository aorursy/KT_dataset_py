# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import stopwords
from textblob import TextBlob
import datetime as dt
import warnings
import string
import time
# stop_words = []
stop_words = list(set(stopwords.words('english')))
warnings.filterwarnings('ignore')
punctuation = string.punctuation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing and setting the database
volcanos = pd.read_csv("../input/volcano_data_2010.csv")
print(volcanos.columns)
volcanos.head(10)
volcanos.info()
volcanos.describe()
# checking missing data
total = volcanos.isnull().sum().sort_values(ascending = False)
percent = (volcanos.isnull().sum()/volcanos.isnull().count()*100).sort_values(ascending = False)
missing_volcanos_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_volcanos_data.head(25)
hold = volcanos["Country"].value_counts().head(30)
hold.iplot(kind='bar', xTitle = 'Country name', yTitle = "# of incidents", title = 'Top countires of Volcanic Activity')
hold = volcanos["Location"].value_counts().head(30)
hold.iplot(kind='bar', xTitle = 'Country name', yTitle = "# of incidents", title = 'Top countires of Volcanic Activity')