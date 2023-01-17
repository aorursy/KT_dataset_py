#https://www.kaggle.com/c/predict-west-nile-virus
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


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
df = pd.read_csv('../input/train.csv')
dg = pd.read_csv('../input/test.csv')
dh = pd.read_csv('../input/spray.csv')
di = pd.read_csv('../input/weather.csv')
df.sample(5)
df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
df.Date.sample(5)
df.Date  = pd.to_datetime(df.Date, format = '%Y-%m-%d')
dg.Date  = pd.to_datetime(dg.Date, format = '%Y-%m-%d')
df.hist()
df.head(10)
sns.lmplot(y='Latitude', x='Longitude', hue='Species', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':.2})
sns.lmplot(y='Latitude', x='Longitude', hue='Species', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':.2})
df.Species.value_counts().sort_index().plot.bar()
dg.Species.value_counts().sort_index().plot.bar()
set(df.columns).difference(set(dg.columns))

DF = df.groupby('Date').NumMosquitos.sum()
#DF = DF.set_index('Date')
DF.plot.line()
# mosquitos occuring every 2 years?