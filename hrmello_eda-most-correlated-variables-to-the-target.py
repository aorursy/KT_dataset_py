import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
chunksize = 100000
train = None

for chunk in pd.read_csv("../input/train.csv", chunksize=chunksize, iterator=True):
    if train is None:
        train=chunk.copy()
    else:
        train.append(chunk)
train.sample(5)
print (train.columns)
train.isnull().sum().sort_values(ascending = False)[:43]
corr_matrix = train.corr()
corr_matrix['HasDetections'].sort_values(ascending = False)
plt.figure(figsize=(20,15))
sb.heatmap(corr_matrix)
plt.title("Correlation Matrix", size = 25)
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

sb.countplot(train.AVProductsInstalled.dropna());

sb.distplot(train["AVProductStatesIdentifier"].dropna(), kde = False);
sb.countplot(train.IsProtected);
sb.countplot(train.Census_IsAlwaysOnAlwaysConnectedCapable.dropna());
sb.distplot(train.Census_TotalPhysicalRAM.dropna(), kde = False, bins = 1000)
plt.xlim(0,30000)
sb.distplot(train.Census_PrimaryDiskTotalCapacity.dropna(), kde = False, bins = 1000)
plt.xlim(0, 1000000)
sb.distplot(train.Census_ProcessorCoreCount.dropna(), kde = False)
plt.xlim(0,10);
sb.countplot(train.Census_IsVirtualDevice.dropna());
sb.countplot(train.Wdft_IsGamer.dropna())
cols_to_use = ['AVProductsInstalled', 'AVProductStatesIdentifier', 
               'Census_IsAlwaysOnAlwaysConnectedCapable','IsProtected',
               'Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity',
               'Census_ProcessorCoreCount', 'Census_IsVirtualDevice', 'Wdft_IsGamer']
plt.figure(figsize=(15,10))
sb.heatmap(train[cols_to_use].corr(), annot = True);