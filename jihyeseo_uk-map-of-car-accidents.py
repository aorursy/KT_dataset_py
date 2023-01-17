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
df = pd.read_csv('../input/acc2005_2016.csv')
df.head()
df.sample(15)
df.Accident_Severity.value_counts()
df.Number_of_Casualties.value_counts()
df.Number_of_Casualties.hist(bins = 50)
df.dtypes
sns.lmplot(y='Latitude', x='Longitude', hue='Accident_Severity', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.01})
sns.lmplot(y='Latitude', x='Longitude', hue='Day_of_Week', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
