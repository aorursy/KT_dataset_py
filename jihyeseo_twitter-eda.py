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
#df = pd.read_csv('../input/gender-classifier-DFE-791531.csv', encoding = 'Windows-1254')

# UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 9: character maps to <undefined>
df = pd.read_csv('../input/gender-classifier-DFE-791531.csv', encoding = 'latin1')


df.shape
df.dtypes
df.link_color.value_counts().plot.bar()
df.gender.value_counts().plot.bar()
df.retweet_count.hist(bins = 30)
# perhaps plot with log scale on y axis
df.tweet_count.hist(bins = 30)
