import pandas as pd

import numpy as np

from pandas import Series, DataFrame, Panel



df = pd.read_csv('../input/Interestrate and inflation Sweden 1908-2001.csv')

%pylab inline


df.tail(20)
df[0:2]
df.shape

# 109 years meaning there is about 3 Kondratiev waves

    # If you aint about the Kondratiev wave theory, then we can't be friends.

    # Anyway, I wish there was recession data... But.. Life goes on...
dates = pd.date_range('1908', '2015')
dates.shape
DF = Series(df[:,2], index=dates)
