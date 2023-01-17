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
#https://stackoverflow.com/questions/44128033/pandas-reading-null-as-a-nan-float-instead-of-str

dfs = dict()
for f in filenames:
    df = pd.read_csv('../input/' + f, thousands = ',', keep_default_na=False, na_values=['NA', '-',''])
    df = df.rename(columns = {'Date(UTC)':'Date'})
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index('Date')
    #print(df.dtypes)
    dfs[f[:-4]] = df
#print(dfs)
dfs['ethereum_dataset'].dtypes
dfs['bitcoin_dataset'].dtypes
dfs['bitcoin_dataset']['btc_market_price'].plot.line()
