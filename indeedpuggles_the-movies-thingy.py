# Phillip McCrevis

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
movies_metadata = pd.read_csv("../input/movies_metadata.csv")
movies_metadata.head()
f,ax = plt.subplots(figsize=(7, 7))
sns.heatmap(movies_metadata.corr(), annot=True, linewidths=.10, fmt= '.1f',ax=ax)
movies_metadata.info()
runtime = movies_metadata.runtime.value_counts()
plt.figure(figsize=(100,30))
sns.barplot(x=runtime[:5].index,y=runtime[:5].values)
runtime.fillna(0)
plt.xticks(rotation=30)
plt.title('Runtime',color = 'blue',fontsize=100)
vote_average = movies_metadata.vote_average.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=vote_average[:10].index,y=vote_average[:10].values)
plt.xticks(rotation=45)
plt.title('Vote_average',color = 'blue',fontsize=15)
movies_metadata[(movies_metadata['vote_average']>=6)]
movies_metadata[(movies_metadata['vote_average']<6)]
sns.set(font_scale=1.25)
cols = ['runtime', 'vote_average']
sns.pairplot(movies_metadata.dropna(how='any')[cols],diag_kind='kde', size = 7.5)
plt.show();