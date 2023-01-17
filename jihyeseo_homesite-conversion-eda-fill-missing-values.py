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
dh = pd.read_csv('../input/sample_submission.csv')
#df.hist()
#dg.hist()
df.sample(5)
df.columns
df.Original_Quote_Date  = pd.to_datetime(df.Original_Quote_Date, format = '%Y-%m-%d')
dg.Original_Quote_Date  = pd.to_datetime(dg.Original_Quote_Date, format = '%Y-%m-%d')
df.describe(exclude = 'O').transpose()
dg.describe(exclude = 'O').transpose()
set(df.columns).difference(set(dg.columns))
# this is the only missing column in test dataset. So this is our target variable
# Let us fill lots of missing variables. 
df.isnull().sum()