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
dw = pd.read_csv("../input/deu_news_2015_3M-words.csv")
dw.head()
dw.head(45)
dw = dw[43:]
dw.head()
dw = dw[['!','53658']]
dw.rename(columns = {'!':'word', '53658':'count'}, inplace = True)
dw.head()
dw.dtypes
len(dw)
dw = dw[:500]
dw
dw['count'] = dw['count'].astype('int64')
 
dw.head(5)


dws = dw.set_index('word').T


dws
dwd = dws.to_dict('list')
dwd
d = {}
for a, x in dwd.items():
    d[a] = x[0]

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
from nltk.corpus import stopwords
stopset = stopwords.words('german')
stopset
wordcloud = WordCloud(stopwords = stopset)
#smalld = dict(filter(lambda i: i not in stopset, d))
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
len(stopset)
wordcloud = WordCloud(stopwords = stopset)
smalld = {k:v for (k,v) in d.items() if k.lower() not in stopset}
wordcloud.generate_from_frequencies(frequencies=smalld)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
