import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_t = pd.read_csv('../input/train.csv').drop(['ID'], axis=1)
plt.figure(figsize=(3,3))
plt.title('target')
df_t['target'].hist()
plt.show()
plt.clf()
plt.close()

df_t = df_t.drop(['target'], axis=1)
df_v = pd.read_csv('../input/test.csv').drop(['ID'], axis=1)
df = pd.concat([df_t,df_v], axis=0, sort=False, ignore_index=True)
columns = df.columns.values
columns.sort()
for i in range(0,columns.shape[0],25):
	plt.figure(figsize=(15,15))
	for j in range(i, min(i+25,columns.shape[0])):
		plt.subplot(5,5,j-i+1)
		plt.title(df.columns[j]+'all/train')
		dd = df[df[columns[j]]!=0][columns[j]]
		mn = dd.min()
		mx = dd.max()
		ax = dd.hist(bins=10, range=(mn,mx))
		df_t[df_t[columns[j]]!=0][columns[j]].hist(ax=ax, bins=10, range=(mn,mx))
	plt.show()
	plt.clf()
	plt.close()
