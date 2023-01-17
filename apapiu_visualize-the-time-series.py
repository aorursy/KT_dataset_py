%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

from pylab import rcParams



from statsmodels.graphics.tsaplots import plot_acf



df = pd.read_csv('../input/air_visit_data.csv', parse_dates=['visit_date'])
rcParams['figure.figsize'] = 16, 300



t = (df.set_index(["air_store_id", "visit_date"])["visitors"]

  .unstack()

  .iloc[:100].T

  .plot(legend=None, subplots = True))

t = (df.set_index(["air_store_id", "visit_date"])["visitors"]

  .unstack()

  .iloc[100:200].T

  .plot(legend=None, subplots = True))
t = (df.set_index(["air_store_id", "visit_date"])["visitors"]

  .unstack()

  .iloc[200:300].T

  .plot(legend=None, subplots = True))