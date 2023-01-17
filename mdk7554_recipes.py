# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

recipes = read_csv("../input/epi_r.csv")
df = recipes.copy(deep=True)
df.head(20)
len(df.columns)
#choose bake as y
df['bake'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='bake',data=df,palette= 'hls')
plt.show()
#check title
df['title'].unique()
# check total number of missing values
df.isnull().sum().sum()
# remove all missing values
df=df.dropna()
df1=df[['title','rating','calories','protein','fat','sodium','bake']]
df1.head()
df1.describe()



df1.groupby('rating').mean()
len(df1)
from patsy import dmatrices
from scipy import stats
y,X = dmatrices('bake ~ rating + calories + protein + fat + sodium', df1, return_type='dataframe')

stats.chisqprob = lambda chisq, df1:stats.chi2.sf(chisq,df1)

#use statsmodel
logit = sm.Logit(y,X)
result = logit.fit()

#print out model
print(result.summary())
print(result.conf_int())
