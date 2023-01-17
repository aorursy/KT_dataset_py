import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
angldata = pd.read_csv('/kaggle/input/ang._mova_11rezult_z_miscyamy_0.txt')
angldata.head()
angldata
angldata.columns
angldata.drop(['номер', 'клас'], axis=1, inplace=True)

angldata.head()
angldata = angldata.sort_values(by='бали')
angldata.describe()
angldata.drop([17], inplace=True)  # got 0, wasn't there
angldata.median()
angldata.mean()
angldata['бали'].plot(kind="bar", stacked='true')
angldata['бали'].hist(alpha=0.5)
#for m in angldata['місто'].unique():

#    angldata[angldata["місто"] == m].boxplot(by="місто")
angldata.boxplot(by="місто", figsize=(35,4))
angldata['імя'] = angldata['піп'].apply(lambda x: x.split()[1])
angldata['імя'].unique()
angldata.boxplot(by="імя", figsize=(20,4))