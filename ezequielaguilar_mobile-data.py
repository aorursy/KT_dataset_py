# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
df_events.head()

print(df_events.describe())
%matplotlib inline
import matplotlib.pyplot as plt
df_events.hist(bins=50, figsize=(11,8))
save_fig("attribute_histogram_plots")
plt.show()