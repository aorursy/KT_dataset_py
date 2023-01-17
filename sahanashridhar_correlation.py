# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

df = pd.read_csv("../input/final-data/out_string.csv")



ax = sns.scatterplot(x="no2_ugm3", y="pm2_5_ugm3", data=df)

ax.invert_yaxis()
corr_NOx = df.corr()['no2_ugm3']



print(corr_NOx)

sns.scatterplot(data = corr_NOx)

plt.xticks(rotation=60)
plt.figure(figsize = (10,8))

sns.heatmap(data=df.corr(),xticklabels=True, yticklabels=True)


print(df.columns)

X = ['NOX_emission',

       'PM_emission', 'ndvi_rvalue', 'road_length_primary_200m',

       'road_count_primary_200m', 'road_length_secondary_200m',

       'road_count_secondary_200m', 'road_length_tertiary_200m',

       'road_count_tertiary_200m', 'road_length_residential_200m',

       'road_count_residential_200m', 'road_length_motorway_200m',

       'road_count_motorway_200m', 'road_length_primary_500m',

       'road_count_primary_500m', 'road_length_secondary_500m',

       'road_length_secondary_500m.1', 'road_length_tertiary_500m',

       'road_count_tertiary_500m', 'road_length_residential_500m',

       'road_count_residential_500m', 'restaurants_500m', 'gas_stations_500m',

       'sub_500m', 'bus_stop_500m']

Y = ['BC', 'NOx']



i=0

for predictor in X[:10]:

    for pollutant in Y:

        plt.figure(i)

        i+=1

        sns.scatterplot(x=predictor, y=pollutant, data=df)
i=0

for predictor in X[10:20]:

    for pollutant in Y:

        plt.figure(i)

        i+=1

        sns.scatterplot(x=predictor, y=pollutant, data=df)
i=0

for predictor in X[20:]:

    for pollutant in Y:

        plt.figure(i)

        i+=1

        sns.scatterplot(x=predictor, y=pollutant, data=df)
import pandas as pd

combined = pd.read_csv("../input/combined.csv")