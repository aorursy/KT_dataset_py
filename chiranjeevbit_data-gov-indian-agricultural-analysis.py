# importing the nessesary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


# Loading datasets
import os
files  = os.listdir("../input")
print(files)

# Any results you write to the current directory are saved as output.
#crop state,crop production, crop varity, crop year, produced
crop_state = pd.read_csv("../input/data 1.csv")
crop_produce = pd.read_csv("../input/data 2.csv")
crop_varity = pd.read_csv("../input/data 3.csv")
crop_year = pd.read_csv("../input/data 4.csv")
produced = pd.read_csv("../input/produced.csv")
crop_state.head()
crop_produce.head()
crop_varity.head()
crop_year.head(12)
produced.head()
crop_produce.columns
k = crop_produce[['Crop             ', 'Production 2006-07', 'Production 2007-08',
       'Production 2008-09', 'Production 2009-10', 'Production 2010-11']].groupby('Crop             ')
index = list(k.indices.keys())
index[:]
# So it is clear that for taking only the total part 
index[-8:-2]
# Now plotting the Year wise production of agricultural crop
k.sum()[:-9].plot(figsize=(20,12), kind='bar');
# -9 for eliminating all the total part
plt.title('Year wise production of agricultural crop')
plt.ylabel('Production in Quintal');

#k.mean().plot(figsize=(20,10), kind='bar');
#plt.figure(figsize=(12,6))
l = len(k['Crop             '])
# l for enumerating throgh all crops
fig, arraxes = plt.subplots(1,4, figsize=(12,12), sharey=True)
plt.setp(arraxes, yticks=range(len(index)), yticklabels = index)

for axes, p in zip(arraxes.flat,['Production 2006-07', 'Production 2007-08','Production 2008-09', 'Production 2009-10', 'Production 2010-11']):
    axes.barh(range(l), k[p].head())
    axes.set_title(p)
#     axes.tick_params(axis='x',  rotation=90)
    axes.set_xlabel("production in Quantal")
fig.set_figwidth(20)
kc = crop_produce[['Crop             ','Area 2006-07', 'Area 2007-08', 'Area 2008-09', 'Area 2009-10',
       'Area 2010-11']].groupby('Crop             ')
kc.sum().plot(figsize=(20,18), kind='barh', stacked= True);


k.head()['Crop             '].values
crop_state.groupby('State').sum()
cols = crop_state.columns
crop_state.groupby('Crop')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));

crop_state.groupby('State')[cols[5:6]].sum().plot(kind='bar', figsize=(12,6));
crop_state.groupby('State')[cols[:-1]].sum().plot(kind='bar', figsize=(12,6));
crop_state.head()
crop_state.groupby('State')[cols[-1]].sum().plot(kind='pie', figsize=(14,14));
plt.title('Crop-wise '+cols[-1], color='red', fontsize=20)
crop_state.groupby('Crop')[cols[-1]].sum().plot(kind='pie', figsize=(20,20));
crop_varity.head()
plt.title('No of varieties per crop in India', fontsize=40, color='orange')
crop_varity['Crop'].value_counts().plot(kind='bar', figsize= (12,12));





