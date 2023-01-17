import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



apple= pd.read_csv("../input/apples-mobility-report/Applemobilitytrends-2020-04-13.csv")



apple.head(2)
apple['geo_type'].unique()
apple['region'].unique()
apple['transportation_type'].unique()
apple1=apple.drop(['geo_type'], axis = 1) 

apple1.head(2)
#transposed the dataset so dates are rows

trans=apple1.T

trans.head(2)
Sea=apple1.loc[apple1['region']== 'Seattle']

Sea.head(2)
Sea=Sea.drop(['region'], axis = 1) 

Sea.head(2)
#transposed data for the city of Seattle

Seattle=Sea.T
#rename and fix dataframe

S = Seattle.rename(columns={348: 'driving', 349: 'transit', 350:'walking'})

S = S.drop(['transportation_type'])

S.head(2)
##change dates to date object

S.index = pd.to_datetime(S.index)

S.head(3)
S.plot.line()