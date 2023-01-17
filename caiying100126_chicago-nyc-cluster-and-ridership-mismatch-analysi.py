import numpy as np
import pandas as pd
import seaborn as sns
df1 = pd.read_csv('chicago_class.csv')
df1.loc[df1.label==1,'importance_by_clustering'] = 0
df1.loc[df1.label==2,'importance_by_clustering'] = 1
df1.loc[df1.label==4,'importance_by_clustering'] = 2
df1.loc[df1.label==3,'importance_by_clustering'] = 3
df1.loc[df1.label==0,'importance_by_clustering'] = 4
df1.importance_by_clustering.value_counts()
binning = pd.cut(df1['ridership-2018'],bins=[0,636590,1286107,2120952,2600000,6604903],
       labels=[4,3,2,1,0]).astype('int32')
df1['importance_by_ridership'] = binning
df1['diff'] = abs(df1['importance_by_clustering']-df1['importance_by_ridership'])
df1.loc[df1['diff']==0].count()
consistency1 = (df1.loc[df1['diff']==0].shape[0])/(df1.shape[0])
print(consistency1)
highdiff1 = (df1.loc[df1['diff']==4].shape[0])/(df1.shape[0])
print(highdiff1)
sns.distplot(df1['diff'],bins=5)
df1.loc[df1['diff']>=3]
df1.loc[df1['diff']>=3].shape[0]
df2 = pd.read_csv('NYC_class.csv')
df2.loc[df2.label==0,'importance_by_clustering'] = 0
df2.loc[df2.label==2,'importance_by_clustering'] = 1
df2.loc[df2.label==1,'importance_by_clustering'] = 2
df2.loc[df2.label==3,'importance_by_clustering'] = 3
df2.importance_by_clustering.value_counts()
binning = pd.cut(df2['ridership'],bins=[0,275166,1115949,3031466,65060657],
       labels=[3,2,1,0]).astype('int32')
df2['importance_by_ridership'] = binning
df2['diff'] = abs(df2['importance_by_clustering']-df2['importance_by_ridership'])
sns.distplot(df2['diff'],bins=4)
df2.loc[df2['diff']>=3]
df2.loc[df2['diff']>=3].shape[0]
consistency2 = (df2.loc[df2['diff']==0].shape[0])/(df2.shape[0])
print(consistency2)
highdiff2 = (df2.loc[df2['diff']==3].shape[0])/(df2.shape[0])
print(highdiff2)