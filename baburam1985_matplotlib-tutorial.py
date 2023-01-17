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

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
regionsDF = pd.read_csv('../input/noc_regions.csv')
athlete_eventsDF = pd.read_csv('../input/athlete_events.csv')
athlete_eventsDF.head()
chnDF = athlete_eventsDF[athlete_eventsDF.Team == 'China']
chnWinningDF = chnDF[~pd.isna(chnDF.Medal) ]
chnWinningDF.head()
chnDF.size
yearwiseData = (chnDF['Year']).value_counts()
yearwiseWinningData = (chnWinningDF['Year']).value_counts()
plt.figure()
plt.scatter(yearwiseData.index,yearwiseData.values)
plt.show()
plt.figure(figsize=(15,10))
plt.bar(yearwiseData.index,yearwiseData.values, width = 1.5)
plt.bar(yearwiseWinningData.index,yearwiseWinningData.values, width = 1.5 )

