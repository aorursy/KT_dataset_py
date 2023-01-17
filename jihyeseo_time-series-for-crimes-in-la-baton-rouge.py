# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


import fuzzywuzzy
from fuzzywuzzy import process

from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input/"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/Baton_Rouge_Crime_Incidents.csv')
df.dtypes
df.head(5)
df['date_parsed'] = pd.to_datetime(df['OFFENSE DATE'], infer_datetime_format=True)
df['date_parsed'].sample(20)
df['CRIME'].value_counts()
df['COMMITTED'].value_counts()
df['date_parsed'].value_counts().sort_values().plot.line()
df['date_parsed'].value_counts().resample('M').sum().plot.line()
df['date_parsed'].value_counts().resample('W').mean().plot.line()
df.groupby('COMMITTED').count()
