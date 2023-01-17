# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.concat([pd.read_csv('../input/esea_master_dmg_demos.part1.csv'), pd.read_csv('../input/esea_master_dmg_demos.part2.csv')])
meta = pd.concat([pd.read_csv('../input/esea_meta_demos.part1.csv'), pd.read_csv('../input/esea_meta_demos.part2.csv')])
meta.groupby('map')['file'].count()
meta = meta[meta['map'] == 'de_mirage'].set_index(['file', 'round'])
df_mirage = df.set_index(['file', 'round']).join(meta, how = 'inner')
df_mirage['s'] = df_mirage['seconds'] - df_mirage['start_seconds']
df_mirage = df_mirage[(df_mirage['s'] < 100) & (df_mirage['s'] > 0)]
df_mirage['s'].hist(bins=100)
df_mirage['bin'] = pd.cut(df_mirage['s'], 20, labels=list(range(0,20))).astype('str')
ddf = df_mirage.groupby(['file', 'round', 'bin', 'att_side', 'vic_side'])[['s', 'att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']].first().dropna().reset_index()

df1 = ddf[ddf['bin'] == '2']
df1['map'] = 'de_mirage'

map_bounds = pd.read_csv('../input/map_data.csv', index_col=0)
md = map_bounds.loc[df1['map']]
md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = (df1.set_index('map')[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']])
md['att_pos_x'] = (md['ResX']*(md['att_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['att_pos_y'] = (md['ResY']*(md['att_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
md['vic_pos_x'] = (md['ResX']*(md['vic_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['vic_pos_y'] = (md['ResY']*(md['vic_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
df1[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']].values

df2 = pd.concat([df1[['att_side', 'att_pos_x', 'att_pos_y']].rename(columns={'att_side':'side', 'att_pos_x':'x', 'att_pos_y':'y'}),
                df1[['vic_side', 'vic_pos_x', 'vic_pos_y']].rename(columns={'vic_side':'side', 'vic_pos_x':'x', 'vic_pos_y':'y'})])
df2['z'] = df2['side'] == 'CounterTerrorist'

from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(100, weights='distance')
clf.fit(df2[['x', 'y']], df2['z'])
xx, yy = np.meshgrid(np.arange(0, 1024, 1),
                     np.arange(0, 1024, 1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt
from scipy.misc import imread

smap = 'de_mirage'

bg = imread('../input/'+smap+'.png')
fig, ax1 = plt.subplots(1,1,figsize=(18,16))
ax1.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax1.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
plt.xlim(0,1024)
plt.ylim(0,1024)
df2[df2['side'] == 'CounterTerrorist'].plot(x='x', y='y', kind='scatter', ax=ax1, color='#3498db', alpha=0.1)
df2[df2['side'] == 'Terrorist'].plot(x='x', y='y', kind='scatter', ax=ax1, color='#f1c40f', alpha=0.1)
ax1.contour(xx, yy, Z, cmap=plt.cm.Paired)


Z
