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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
df = pd.read_csv('../input/screening_times.csv')
dg = pd.read_csv('../input/TOF_CO2_data_30sec.csv')
df.shape
dg.shape
df.head()
dg.head()
dg.dtypes.head()
df.dtypes
pd.to_datetime(dg.Time, format = '%m/%d/%Y %H:%M:%S')
dg['Time_parsed'] = pd.to_datetime(dg.Time, infer_datetime_format=True)
dg = dg.set_index('Time_parsed')
dg.columns
dg.CO2.plot()
dg[dg.columns[2:5]].plot()
df.head()
df['begin_parsed'] = pd.to_datetime(df['begin'], infer_datetime_format = True)
df.head()
df = df.set_index('begin_parsed')
df.head()
df['number visitors'].plot.line()
df['number visitors'].resample('D').sum().plot.line()
df['number visitors'].resample('W').sum().plot.line()
df['filled %'].plot.line()
df.head()
df.groupby('movie')['number visitors'].sum().sort_values(ascending = False)
