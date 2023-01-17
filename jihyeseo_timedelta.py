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
dg = pd.read_csv('../input/glasses.csv')
dr = pd.read_csv('../input/report.csv')
dp = pd.read_csv('../input/smartphone.csv')
dw = pd.read_csv('../input/smartwatch.csv')
dg.head()
dr.head()
dp.head()
dw.head()
dw.dtypes
# https://pandas.pydata.org/pandas-docs/stable/timedeltas.html 
#pd.Timedelta('00:' + dr.duration.dropna().astype(str))
pd.Timedelta('00:00:20') + pd.Timedelta('00:00:20')
#pd.Timedelta('00:' + dr.duration)
dr.duration.unique()
 #pd.Timedelta('00:' + dr.duration.dropna().astype(str))
dr['fullDuration'] = '00:' + dr.duration.astype(str)
#pd.to_timedelta(dr.fullDuration.str, unit = 's')
len(dr)
