import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
sellp = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
ss = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
stv = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
stv.head()
d_cols = [c for c in stv.columns if 'd_' in c]
stv.head()
print(stv['id'].describe())
cal.head()

