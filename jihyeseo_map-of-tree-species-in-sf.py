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
df = pd.read_csv('../input/san_francisco_street_trees.csv')
df.head()
df.dbh.plot.hist()
df.latitude.plot.hist()
df.longitude.plot.hist()
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='plant_type', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.2})
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='legal_status', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.2})
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='care_assistant', 
           data=df, 
           fit_reg=False, scatter_kws={'alpha':0.2})
df.dtypes
df[['care_assistant', 'care_taker',  
       'legal_status', 'permit_notes',  
       'plant_type',  'site_info', 'species']].T.apply(lambda x: x.nunique(), axis=1).sort_values()
df.columns
df.plant_type.value_counts()
# certainly this needs some cleaning from tree to Tree.
df.legal_status.value_counts()
# what does 'Significant Tree                 125' mean?
# Landmark tree  ?
df.care_assistant.value_counts()
#df.species.value_counts()
# Want to explore cherry trees, magnolia, pine tree etc, chestnut, beech, cedar, olive, acacia
df['cleanname'] = df.species.str.lower().str.strip()
df['firstname'] = df.species.str.split().str.get(0)

vc = df.firstname.value_counts()
mycat = vc.index[:5].tolist()
dg = df[(df.firstname.isin(mycat)) & (df.latitude < 40) & (df.longitude > - 126) & (df.y_coordinate > 2080000)]
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='firstname', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='legal_status', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='care_assistant', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})
sns.lmplot(x='x_coordinate', y='y_coordinate', hue='plant_type', 
           data=dg, 
           fit_reg=False, scatter_kws={'alpha':0.1})
