import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')

import gc





# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import matplotlib.patches as patches

# Suppress warnings 

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
import os

print(os.listdir("../input/cusersmarildownloadsdeathscsv/"))
%%time

df = pd.read_csv('../input/cusersmarildownloadsdeathscsv/deaths.csv', delimiter=';', encoding = "ISO-8859-1")

df.dataframeName = 'deaths.csv'
print('Size of df data', df.shape)
df.head()
total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(8)
# Number of each type of column

df.dtypes.value_counts()
# Number of unique classes in each object column

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
corrs = df.corr()

corrs
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
HTML('<iframe width="980" height="520" src="https://www.youtube.com/embed/PL2cXmFL-OM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
HTML('<iframe width="980" height="520" src="https://www.youtube.com/embed/onFx-7b2M3I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
HTML('<iframe width="980" height="520" src="https://www.youtube.com/embed/VSyj_OC4QO4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')