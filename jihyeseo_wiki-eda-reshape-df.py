#https://www.kaggle.com/c/web-traffic-time-series-forecasting/data

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
df1 = pd.read_csv('../input/train_1.csv')
dg1 = pd.read_csv('../input/sample_submission_1.csv')
dh1 = pd.read_csv('../input/key_1.csv')
df2 = pd.read_csv('../input/train_2.csv')
dg2 = pd.read_csv('../input/sample_submission_2.csv')
dh2 = pd.read_csv('../input/key_2.csv')
df1.sample(5)
df1.shape
dh1.sample(5)
dg1.sample(5)
