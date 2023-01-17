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
df = pd.read_csv('../input/pax_20_02_2018_1_CSV.csv')
df.head()
df.Reg.value_counts()
df.Con.value_counts()
sns.countplot(x = 'Reg', data = df)
sns.countplot(x="Reg", data=df,
              order=df.Reg.value_counts().iloc[:5].index)
sns.countplot(x="Con", data=df,
              order=df.Con.value_counts().iloc[:10].index)
sns.countplot(x="Con", data=df,
              order=df.Con.value_counts().iloc[:5].index)
df['date'] = pd.to_datetime(df.Dat, infer_datetime_format= True)

df.date.sample(20)
df['date'].value_counts().resample('M').sum().plot.line()
df['date'].value_counts().resample('Y').sum().plot.line()
# I want time series per each categories - how to do this efficiently in python?

