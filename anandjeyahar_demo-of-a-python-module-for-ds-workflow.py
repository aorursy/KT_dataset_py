# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Custom libraries

from datascienceutils import plotter

from datascienceutils import analyze





# Standard libraries

get_ipython().magic('load_ext autoreload')

#import matplotlib as mpl

#mpl.use('Agg')

#import matplotlib.pyplot as plt

import json

#fig=plt.figure()

get_ipython().magic('matplotlib inline')



import numpy as np

import pandas as pd

from sklearn import cross_validation

from sklearn import metrics





from bokeh.plotting import figure, show, output_file, output_notebook, ColumnDataSource

from bokeh.charts import Histogram

import bokeh

output_notebook(bokeh.resources.INLINE)



from sqlalchemy import create_engine

df = pd.read_csv('../input/Iris.csv')
df.describe()

df.head()

df.corr()
analyze.correlation_analyze(df, exclude_columns='Id')
df.columns

target = df.Species

df.drop('Species', 1, inplace=True)

analyze.silhouette_analyze(df, cluster_type='KMeans', n_clusters=range(2,4))

## Now that's some magic and a few more (clustering) algorithm varieties below
analyze.silhouette_analyze(df, cluster_type='spectral', n_clusters=range(2,5))

analyze.cluster_analyze(df, cluster_type='KMeans', n_clusters=4)



analyze.cluster_analyze(df, cluster_type='dbscan')

analyze.som_analyze(df, (30,30), algo_type='som')
## If you're impressed look here https://github.com/greytip/-data-science-utils or just do `pip install datascienceutils` and enjoy

## Caveat: Not much documentation available at the moment, so will need to read code ,and documentation of original libraries.

## In any case leave feedback on the github issues page.
