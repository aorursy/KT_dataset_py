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
df = pd.read_csv('../input/KAG_NCHS_-_Births__Birth_Rates__and_Fertility_Rates__by_Race_of_Mother__United_States.csv')
df.head()
df.dtypes
df = df.set_index('Year')
df['Live Births'].plot.line()
from pandas.plotting import parallel_coordinates
 
parallel_coordinates(df, 'Race')
df.tail()
df.Race.unique()
df.groupby('Race')['Live Births'].sum()
df = df.reset_index()
df
sns.lmplot(x='Year', y='Live Births', hue='Race', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.8})
sns.lmplot(x='Year', y='Birth Rate', hue='Race', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.8})
sns.lmplot(x='Year', y='Fertility Rate', hue='Race', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.8})
