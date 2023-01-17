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
df = pd.read_csv('../input/' + filenames)
print(df.dtypes)
print(df.shape)
df.head()
print(df.isnull().sum())
df.describe(exclude = 'O').transpose()


print(df['User_Score'].unique())
df['User_Score'] = pd.to_numeric(df.User_Score, errors = 'coerce')

df['Rating'].unique()
df.describe(include = 'O').transpose()
df.Rating.value_counts().plot.bar()
df.columns
sns.lmplot(y='NA_Sales', x='Year_of_Release', hue = 'Rating' ,
           data=df, 
           fit_reg=False, scatter_kws={'alpha':1})
sns.lmplot(y='EU_Sales', x='Year_of_Release', hue = 'Rating' ,
           data=df, 
           fit_reg=False, scatter_kws={'alpha':1})
sns.lmplot(y='User_Count', x='Year_of_Release', hue = 'Rating' ,
           data=df, 
           fit_reg=False, scatter_kws={'alpha':1})
sns.lmplot(y='User_Score', x='Year_of_Release', hue = 'Rating' ,
           data=df, 
           fit_reg=False, scatter_kws={'alpha':1})
df.isnull().sum()
