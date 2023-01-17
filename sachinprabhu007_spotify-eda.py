import warnings

warnings.filterwarnings('ignore')



import pandas_profiling as pd_prof

import numpy as np

import pandas as pd



import seaborn as sns



sns.set()
import matplotlib.pyplot as plt

import os

print(os.listdir('../input/'))
music_data = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
music_data.shape
pd_prof.ProfileReport(music_data)
music_data.info()
music_data.head()
music_data.describe()
m = music_data.hist(figsize=(20,20))
music_data['Genre'].value_counts()
plt.figure(figsize=(20,20)) 

m=sns.heatmap(music_data.corr(), annot=True,cmap ='RdYlGn') 