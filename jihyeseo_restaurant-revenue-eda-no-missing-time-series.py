#https://www.kaggle.com/c/homesite-quote-conversion
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
dh = pd.read_csv('../input/sampleSubmission.csv')
di = pd.read_csv('../input/leaderboard.csv')
set(df.columns).difference(set(dg.columns))
df.isnull().sum().sum()
dg.isnull().sum().sum()
df['Open Date'].sample(5)
df['Open Date']  = pd.to_datetime(df['Open Date'], format = '%m/%d/%Y')
dg['Open Date']  = pd.to_datetime(dg['Open Date'], format = '%m/%d/%Y')
df.columns
df.describe(include = 'O').transpose()
dg.describe(include = 'O').transpose()
df['Type'].value_counts().sort_index().plot.bar()
dg['Type'].value_counts().sort_index().plot.bar()
df['City Group'].value_counts().plot.bar()
dg['City Group'].value_counts().plot.bar()
df.describe(exclude = 'O').transpose()
dg.describe(exclude = 'O').transpose()
df.isnull().sum()
df.P1.describe()
DF = df.set_index('Open Date')
DF['revenue'].plot.line()
