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
for size in range(1,10):
    length = 10 ** size
    with open("../input/" + filenames, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)
df = pd.read_csv("../input/" + filenames, sep='\t', error_bad_lines=False, warn_bad_lines = True)

df.head()
df.Language.unique()
#df.Date = 
df.Date = pd.to_datetime(df.Date, format = '%Y/%m/%d', errors = 'coerce' )
korean = df[df.Language == 'Korean']
german = df[df.Language == 'German']
japanese = df[df.Language == 'Japanese']
korean.head()
german.head()
japanese.head()
dg = df.groupby(['Language','Source','Date']).Text.count()
dg = dg.reset_index()
korean = dg[dg.Language == 'Korean']
german = dg[dg.Language == 'German']
japanese = dg[dg.Language == 'Japanese']
koreanSources = korean.groupby('Source').Text.sum().sort_values(ascending = False)
koreanSources.head(5).plot.bar()

german.groupby('Source').Text.sum().sort_values(ascending = False).head(5).plot.bar()
japanese.groupby('Source').Text.sum().sort_values(ascending = False).head(5).plot.bar()
