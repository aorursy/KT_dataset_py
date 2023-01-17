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
filenames = filenames.split('\n')
dfs = []
for filename in filenames:
    df = pd.read_csv('../input/' + filename)
    print(df.head())
    dfs.append(df)

dfs[0].sample(5)
dfs[5].sample(5)
dfs[0].category.unique()
dfs[10].sample(10)
dfs[11].sample(10)
dfs[10].category.unique()
dfs[11].category.unique()
dfs[10].sort_values(by = 'value', ascending = False).head(10)
dfs[11].sort_values(by = 'value', ascending = False).head(10)
filenames[10]
highGDP = dfs[10].groupby('country_or_area').value.sum().sort_values(ascending = False).head(10)
richCountries = highGDP.index.tolist()
type(richCountries)
table2 = pd.pivot_table(dfs[10], values='value', index=['year'], columns=['country_or_area'])
table2.head()
table2[richCountries].plot()
