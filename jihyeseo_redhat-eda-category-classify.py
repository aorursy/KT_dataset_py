#https://www.kaggle.com/c/predicting-red-hat-business-value
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
df = pd.read_csv('../input/act_train.csv')
dg = pd.read_csv('../input/act_test.csv')
dh = pd.read_csv('../input/sample_submission.csv')
di = pd.read_csv('../input/people.csv')
df.sample(5)
di.sample(5)
df.describe(include = 'O').transpose()
dg.describe(include = 'O').transpose()
df.describe(exclude = 'O').transpose()
df.date  = pd.to_datetime(df.date, format = '%Y-%m-%d')
dg.date  = pd.to_datetime(dg.date, format = '%Y-%m-%d')
DF = df.set_index('date')
DF['outcome'].plot.line()
df.hist()
