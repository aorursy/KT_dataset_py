# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


import fuzzywuzzy
from fuzzywuzzy import process

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
df = pd.read_csv('../input/Seattle_Police_Department_911_Incident_Response.csv')
df.head()
sns.lmplot(x='Longitude', y='Latitude', hue='District/Sector', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='Longitude', y='Latitude', hue='Zone/Beat', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='Longitude', y='Latitude', hue='Event Clearance Group', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.1})
