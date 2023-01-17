import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Image('../input/structure-photos/Overall photo.JPG', width = 400, height = 400)
traindf = pd.read_csv('/kaggle/input/cee-498-project9-structural-damage-detection/train.csv')

traindf.dropna(inplace = True)

traindf.reset_index(inplace = True)

traindf.drop(labels ='index',axis=1,inplace=True)
traindf.shape
traindf.dtypes
traindf.describe()
plt.figure(figsize = (12,12))

plt.title('Correlation between all sensors and condition')

sns.heatmap(data = traindf.corr(),

            annot = True,

            cmap = 'copper_r',

            square = True,

            linewidths = 1);
traindf.hist(figsize=(20,20))
var, var2 = 'DA04','DA07'

data = pd.concat([traindf[var2], traindf[var]], axis=1)

data.plot.scatter(x=var, y='DA07');
undamaged = traindf.loc[traindf['Condition'] == 1]

damaged = traindf.loc[traindf['Condition'] == 0]
plt.figure(figsize = (12,12))

plt.title('Correlation between all sensors and condition')

sns.heatmap(data = undamaged.corr(),

            annot = True,

            cmap = 'copper_r',

            square = True,

            linewidths = 1);
plt.figure(figsize = (12,12))

plt.title('Correlation between all sensors and condition')

sns.heatmap(data = damaged.corr(),

            annot = True,

            cmap = 'copper_r',

            square = True,

            linewidths = 1);
undamaged.hist(figsize=(20,20))
damaged.hist(figsize=(20,20))
plt.figure(figsize=(30,4))

plt.plot(traindf.index,traindf['DA04'])
var, var2 = 'DA04','DA07'

data = pd.concat([undamaged[var2], undamaged[var]], axis=1)

data.plot.scatter(x=var, y='DA07');
var, var2 = 'DA04','DA07'

data = pd.concat([damaged[var2], damaged[var]], axis=1)

data.plot.scatter(x=var, y='DA07');