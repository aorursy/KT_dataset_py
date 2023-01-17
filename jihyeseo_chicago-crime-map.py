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
filenames = filenames.split('\n')

#   df = pd.read_csv('../input/' + f, thousands = ',', keep_default_na=False, na_values=['NA', '-',''])
dflist = []
for f in filenames:
    print(f)
    df = pd.read_csv('../input/' + f, error_bad_lines=False, warn_bad_lines = True)
    #https://stackoverflow.com/questions/1759455/how-can-i-account-for-period-am-pm-with-datetime-strptime
    df['Updated On'] = pd.to_datetime(df['Updated On'], format = '%m/%d/%Y %I:%M:%S %p', errors = 'coerce')
    df.Date = pd.to_datetime(df.Date, format = '%m/%d/%Y %I:%M:%S %p', errors = 'coerce')
    dflist.append(df)
    print(df.shape)
df = pd.concat(dflist)
df.dtypes
df['Primary Type'].unique()
df.Latitude = pd.to_numeric(df.Latitude, errors = 'coerce')
#   df = pd.read_csv('../input/' + f, thousands = ',', keep_default_na=False, na_values=['NA', '-',''])
sns.lmplot(x='Longitude', y='Latitude',# hue='Primary Type', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})

# there seems natural clustering of two locations. Why so? Where are these?
sns.lmplot(x='Longitude', y='Latitude', hue='Primary Type', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
