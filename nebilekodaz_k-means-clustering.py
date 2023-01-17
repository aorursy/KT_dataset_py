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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/College.csv',index_col=0)
df.head()
df.info()
df.describe()
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

plt.figure(figsize=(15,8))
df[df['Private']=='Yes']['Outstate'].hist(alpha=0.5,color='blue',
                                              bins=20,label='Pirvate Yes')
df[df['Private']=='No']['Outstate'].hist(alpha=0.5,color='red',
                                              bins=20,label='Private No')
plt.legend()
plt.xlabel('Outstate')

plt.figure(figsize=(15,8))
df[df['Private']=='Yes']['Grad.Rate'].hist(alpha=0.5,color='blue',
                                              bins=20,label='Pirvate Yes')
df[df['Private']=='No']['Grad.Rate'].hist(alpha=0.5,color='red',
                                              bins=20,label='Private No')
plt.legend()
plt.xlabel('Grad.Rate')
df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate'] > 100]

plt.figure(figsize=(15,8))
df[df['Private']=='Yes']['Grad.Rate'].hist(alpha=0.5,color='blue',
                                              bins=20,label='Pirvate Yes')
df[df['Private']=='No']['Grad.Rate'].hist(alpha=0.5,color='red',
                                              bins=20,label='Private No')
plt.legend()
plt.xlabel('Grad.Rate')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop(['Private'], axis = 1))
kmeans.cluster_centers_
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
df['Cluster'] = df['Private'].apply(converter)
df.head()
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))
