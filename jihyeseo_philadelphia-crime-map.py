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
df.dtypes
df.Dispatch_Date_Time  = pd.to_datetime(df.Dispatch_Date_Time, format = '%Y-%m-%d %H:%M:%S')
#df.sample(5)
df.drop(columns = ['Dispatch_Time','Dispatch_Date','Month'], inplace = True)
df.dtypes
df.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
dH = df.groupby('Hour').Dispatch_Date_Time.count()
dH.sort_index().plot.bar()
# 5-6 hours, the least number of calls
sns.lmplot(x='Lon', y='Lat',# hue='Zone/Beat', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.7})
print(df.Psa.value_counts())
df.Text_General_Code.value_counts()
sns.lmplot(x='Lon', y='Lat',  hue='Text_General_Code', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.7})
