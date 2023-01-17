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
df = pd.read_csv('../input/train.csv')
dg = pd.read_csv('../input/test.csv')
dh = pd.read_csv('../input/sample_submission.csv')
di = pd.read_csv('../input/stores.csv')
dj = pd.read_csv('../input/oil.csv')
dk = pd.read_csv('../input/items.csv')
dl = pd.read_csv('../input/holidays_events.csv')
dm = pd.read_csv('../input/transactions.csv') 
df.date = pd.to_datetime(df.date, format = '%Y-%m-%d')
dg.date = pd.to_datetime(dg.date, format = '%Y-%m-%d')
dj.date = pd.to_datetime(dj.date, format = '%Y-%m-%d')
dl.date = pd.to_datetime(dl.date, format = '%Y-%m-%d')
dm.date = pd.to_datetime(dm.date, format = '%Y-%m-%d') 
dm.head()
dm.groupby('date').transactions.sum().plot.line()
dl.head()
dk.head()
dk['class'].value_counts().plot.bar()
dk.family.value_counts().plot.bar()
dj.head()
di.head()
di.describe(include = 'O').transpose()
di.type.value_counts().plot.bar()
dh.head()
dg.head()
df.head()

