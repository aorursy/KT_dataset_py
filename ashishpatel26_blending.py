import numpy as np # NUMPY
import pandas as p # PANDAS

# DATA VIZUALIZATION LIBRARIES
from matplotlib import pyplot as plt
import seaborn as sns

# METRICS TO MEASURE RMSE
from math import sqrt
from sklearn import metrics
df_base0 = p.read_csv('../input/1a2a3a/final_submission/1a.csv',names=["ID","target"], skiprows=[0],header=None)
df_base1 = p.read_csv('../input/1a2a3a/final_submission/2a.csv',names=["ID","target"], skiprows=[0],header=None)
df_base2 = p.read_csv('../input/1a2a3a/final_submission/3a.csv',names=["ID","target"], skiprows=[0],header=None)
df_base = p.merge(df_base0,df_base1,how='inner',on='ID')
df_base = p.merge(df_base,df_base2,how='inner',on='ID')
#CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
plt.figure(figsize=(20,20))
sns.heatmap(df_base.iloc[:,1:].corr(),annot=True,fmt=".2f")
#ALTERNATIVE WAY - RMSE MATRIX (RMSE to measure how similar are 2 solutions)
M = np.zeros([df_base.iloc[:,1:].shape[1],df_base.iloc[:,1:].shape[1]])
for i in np.arange(M.shape[1]):
    for j in np.arange(M.shape[1]):
        M[i,j] = sqrt(metrics.mean_squared_error(df_base.iloc[:,i+1], df_base.iloc[:,j+1]))
plt.figure(figsize=(20,10))
sns.heatmap(M,annot=True,fmt=".3f")
#SOLUTION = MEAN OF COLUMNS
df_base['target'] = df_base.iloc[:,1:].mean(axis=1)
df_base.head()
#GENERATING FINAL SOLUTION
df_base[['ID','target']].to_csv("latest.csv",index=False)
