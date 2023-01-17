import os
os.chdir('/kaggle/input/red-wine-quality-cortez-et-al-2009/')
print(os.getcwd())
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# data import and view
df = pd.read_csv('winequality-red.csv')
print(df.head(),df.shape)
#### 
# label the wines:
# let's say bad wines are scored 1-3, fair wines are 4-6, good wines are 7-9

df['grade'] = ['bad' if (x >= 1) & (x <= 3) else 'fair' if (x >= 4) & (x <= 6) else 'good' for x in df['quality']] 
print(df)
# features correlation with quality
print(df.corr()['quality'])
train_features = df.columns[:-2]

for i in train_features:
    plt.show(sn.violinplot(df['grade'],df[i],order=['bad','fair','good'],palette='muted',inner="quartile"))
    plt.show()
print('mean of criteria for good, fair and bad wines:')
bad_col = df[df['grade']=='bad'].mean().round(3)
fair_col = df[df['grade']=='fair'].mean().round(3)
good_col = df[df['grade']=='good'].mean().round(3)
df_mean = pd.DataFrame([bad_col,fair_col,good_col],index=['bad_col','fair_col','good_col']).T
print(df_mean)
km = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=20, 
               tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
df['result'] = km.fit_predict(df[train_features])
compare = df[['result','quality']]
compare.groupby('result').hist()
