import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt
data = pd.read_csv('../input/Indicators.csv')

data.shape
data.describe()
data.head()
indicators = data['IndicatorName'].unique().tolist()

len(indicators)
#indicators
cc = data['CountryName'].str.contains('India')
consumptions= data['IndicatorName'] == 'GNI (constant LCU)'

stage = data[consumptions&cc]

stage['IndicatorName'].unique().tolist()
mechandise =  data['IndicatorName'].str.contains('Merchandise exports')


stage = data[consumptions&cc]

len(stage)

stage['IndicatorName'].unique().tolist()
stage2 = data[mechandise&cc]

#v = list('Merchandise exports (current US$)')

xxx = (stage2['IndicatorName'] == 'Merchandise exports (current US$)')

stage2  = stage2[xxx]
len(stage2)
stage2['IndicatorName'].unique().tolist()
import matplotlib.pyplot as plt



fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('exports vs. GNI )',fontsize=10)

axis.set_xlabel(stage['IndicatorName'].iloc[0],fontsize=10)

axis.set_ylabel(stage2['IndicatorName'].iloc[0],fontsize=10)



X = stage['Value']

Y = stage2['Value']



axis.scatter(X, Y)

plt.show()
stage['IndicatorName'].iloc[0]
