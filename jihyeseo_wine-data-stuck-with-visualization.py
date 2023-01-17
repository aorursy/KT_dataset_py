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
df = pd.read_csv('../input/majestic_wine_data.csv', encoding = 'latin1')
df.head()
df.wine_type.value_counts().plot.bar()
df.shape
# sns.pairplot(df[['num_ratings','positive_rating','price','rating_score']], dropna =True)
#https://stackoverflow.com/questions/31493446/seaborn-pairplot-and-nan-values 
# ValueError: max must be larger than min in range parameter.
df.hist()
sns.lmplot(x='price', y='rating_score',  hue='wine_type', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.7})

sns.boxplot(x='price', y='rating_score',  hue='wine_type', data=df)
from pandas.plotting import parallel_coordinates
 
#df.head()
parallel_coordinates(df, 'wine_type') #, color = ['brown','green'] )
