# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
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

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/' + filenames)
df.head()
df.dtypes
df.callDateTime = pd.to_datetime(df.callDateTime , format = '%Y-%m-%d %H:%M:%S')
df.head(1).location.str.strip('()').str.split(',')
def latlon(st):
    stuff = (st.strip('()').split(','))
    if len(stuff) == 2:
        la = stuff[0]
        lo = stuff[1]
        if len(la) == 0 or abs(float(la)) > 1000 or abs(float(la)) < 1 : 
            la = np.nan
        else:
            la = float(la)
        if len(lo) == 0 or abs(float(lo)) > 1000 or  abs(float(lo)) < 1: 
            lo = np.nan
        else:
            lo = float(lo)
                        
        return la, lo
    else:
        return np.nan, np.nan
df["lat"], df["lon"] = zip(*df["location"].map(latlon))
df.lat.describe()
df.lon.describe()
df.lat.hist(bins = 50)
df.lon.hist(bins = 50)
df.district.value_counts().plot.bar()
sns.lmplot(x='lon', y='lat', hue='district', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='lon', y='lat', hue='priority', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.3})
