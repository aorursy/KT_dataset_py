# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
tedt = pd.read_csv('../input/ted_main.csv')
#read dataset
tedt.head()
print(tedt.info(verbose=False))
import missingno as msno
msno.matrix(tedt.sample(500))
#import plot libraries for interacting visuals
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
# For Notebooks
init_notebook_mode(connected=True)
# For offline use
cf.go_offline()
mostviews = tedt[['title', 'main_speaker','speaker_occupation','views', 'languages', 'duration']].sort_values('views', ascending=False)
mostviews.head()
tedt[tedt['languages']==72]
tedt['languages'].iplot(kind='hist', xTitle='Number of Languages', yTitle='Number of Talks')

mostviews.iplot(kind='scatter', x='languages', y='views', mode='markers')
publish_date = pd.to_datetime(tedt["published_date"], unit='s').dt.year
popularity = pd.DataFrame(tedt.groupby(publish_date)['views', 'languages', 'comments'].mean())
popularity
tedt['speaker_occupation'].value_counts().head()

