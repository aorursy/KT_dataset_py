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
df = pd.read_csv('../input/menu.csv')
df.shape
df.dtypes
df['Serving Size'].value_counts().plot.bar()
df['Item'].value_counts().plot.bar()
df['Item'].unique()
# Could do some data cleaning of the item name, to make category values, with fuzzy text etc
# or could do natural clustering, unsupervised learning
sns.lmplot(y='Trans Fat', x='Cholesterol',  hue = 'Category',
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.7})
df.columns
df.Category.value_counts().plot.bar()
sns.lmplot(y='Total Fat', x='Sugars', hue = 'Category' ,
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.7})
