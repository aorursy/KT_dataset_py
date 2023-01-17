# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.mode.chained_assignment = None

df_train = pd.read_csv('../input/train.csv', nrows=1000000)
print('The coulmn List : {}'.format(df_train.columns.tolist()))

column = df_train.columns.tolist()[1:-3]
y = 'is_attributed'
#tmp_group = [train_1.groupby([col,y]).size() for col in column]
f, ax = plt.subplots(1,4, figsize = (12,4))
list_major_group = []
for i,col in enumerate(column):
    tmp = df_train.groupby([col,y]).size().reset_index()
    tmp.rename(columns = {0:'cnt'}, inplace = True)
    tmp = tmp.pivot(index = col, columns = 'is_attributed', values = 'cnt')
    tmp_index = np.full(tmp.shape[0], True, dtype = bool)
    for num in [0,1]:
        tmp_index &= ~tmp[num].isnull()
    tmp = tmp.loc[tmp_index,:]
    tmp['sum'] = tmp.sum(axis = 1)
    tmp['ratio_1'] = tmp[1] / tmp['sum']
    tmp = tmp.loc[(tmp['ratio_1'] > 0.1) & (tmp['sum'] > 50),:]
    tmp = tmp.sort_values('ratio_1', ascending = False).head(20)
    tmp.sort_values('sum', inplace = True)
    list_major_group.append(tmp.index)
    tmp[[0,1]].plot.barh(stacked = True, ax = ax[i], legend = False)
    title = str(col)
    ax[i].set_title(title)
    ax[i].set_ylabel('')
ax[0].set_ylabel('Column Value')
ax[i].legend()
plt.subplots_adjust(wspace = 0.2, top = 0.85)
plt.suptitle('Stacked Bar is_attribued Groupby Column', size = 14)
plt.show()
from itertools import combinations
for grp1, grp2 in combinations(list_major_group,2):
    grp1_name, grp2_name = grp1.name, grp2.name
    grp1_val, grp2_val = grp1.values, grp2.values
    grp1_tf = df_train[grp1_name].isin(grp1_val)
    grp2_tf = df_train[grp2_name].isin(grp2_val)
    grpU_tf = grp1_tf & grp2_tf
    print('{0} : {1} ({2:0.1f})'.format(grp1_name, grp2_name, (grp1_tf & grp2_tf).sum() / (grp1_tf | grp2_tf).sum()))
kernel = df_train[['device', 'is_attributed']].copy()
kernel = kernel.groupby(['device', 'is_attributed']).size()
kernel = kernel.reset_index()
kernel.rename(columns = {0:'cnt'}, inplace = True)
kernel = kernel.pivot(index = 'device', columns = 'is_attributed', values = 'cnt')
kernel.fillna(0, inplace = True)
kernel['sum'] = kernel.sum(axis=1)
kernel['ratio'] = kernel[1].divide(kernel['sum']).round(2)

plt.figure(figsize = (12,12))
ax = plt.subplot2grid((3, 4), (0, 0))
height_ratio = [(kernel['ratio']==0).sum(), (kernel['ratio']!=0).sum()]
ax.bar(x = [0, 1], height = height_ratio, color = ['green', 'red'])
ax.set_xticks([0, 1])

for i, height in enumerate(height_ratio):
    ax.text(i, height+2, height, ha = 'center', color = 'grey')
ax.set_ylim([0,max(height_ratio)+10])
ax.set_xlabel('Is_attribued')
ax.set_title('The # Device')

tmp_index = kernel.index[(kernel['ratio'] != 0) & (10 < kernel['sum']) & (kernel['sum'] < 50)]
kernel = kernel.loc[tmp_index, ['sum', 'ratio']]
kernel = kernel.reset_index()
kernel.rename(columns = {'index': 'device'})
kernel.sort_values('ratio', ascending = False, inplace = True)
kernel.reset_index(drop = True, inplace = True)
kernel.reset_index(inplace = True)

ax = plt.subplot2grid((3, 4), (0, 1), colspan = 3)
sns.barplot(x=kernel.index, y="ratio", data=kernel, palette = sns.color_palette("Blues_d", kernel.shape[0]), ax = ax)
#ax.barh(effect_channel.index, effect_channel.ratio,align = 'center', color = 'green')
ax.set_xticks(kernel.index)
ax.set_xticklabels(kernel.device)
#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlim([0,0.5])
ax.set_ylabel('Performance')
ax.set_xlabel('Device')
ax.set_title('Devicel Performance')
ax.set_ylim([0,0.5])
for i, text in enumerate(kernel.ratio):
    if text < 0.1: break
    ax.text(i, text+0.02, text,ha = "center", color = 'grey', fontsize = 8)
ax.hlines(0.1, 0, kernel.shape[0], color = 'red')
ax.set_title('Device Performance')
ax.set_ylabel('')
effect_device = kernel['device'].loc[kernel.ratio >= 0.1].tolist()


kernel = df_train[['app', 'is_attributed']].copy()
kernel = kernel.groupby(['app', 'is_attributed']).size()
kernel = kernel.reset_index()
kernel.rename(columns = {0:'cnt'}, inplace = True)
kernel = kernel.pivot(index = 'app', columns = 'is_attributed', values = 'cnt')
kernel.fillna(0, inplace = True)
kernel['sum'] = kernel.sum(axis=1)
kernel['ratio'] = kernel[1].divide(kernel['sum']).round(2)


ax = plt.subplot2grid((3, 4), (1, 0))
height_ratio = [(kernel['ratio']==0).sum(), (kernel['ratio']!=0).sum()]
ax.bar(x = [0, 1], height = height_ratio, color = ['green', 'red'])
ax.set_xticks([0, 1])

for i, height in enumerate(height_ratio):
    ax.text(i, height+2, height, ha = 'center', color = 'grey')
ax.set_ylim([0,max(height_ratio)+10])
ax.set_xlabel('Is_attribued')
ax.set_title('The # App')

tmp_index = kernel.index[(kernel['ratio'] != 0) & (10 < kernel['sum']) & (kernel['sum'] <= 50)]
kernel = kernel.loc[tmp_index, ['sum', 'ratio']]
kernel = kernel.reset_index()
kernel.rename(columns = {'index': 'app'})
kernel.sort_values('ratio', ascending = False, inplace = True)
kernel.reset_index(drop = True, inplace = True)
kernel.reset_index(inplace = True)


ax = plt.subplot2grid((3, 4), (1, 1), colspan = 3)
sns.barplot(x=kernel.index, y="ratio", data=kernel, palette = sns.color_palette("Blues_d", kernel.shape[0]), ax = ax)
#ax.barh(effect_channel.index, effect_channel.ratio,align = 'center', color = 'green')
ax.set_xticks(kernel.index)
ax.set_xticklabels(kernel.app)
#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlim([0,0.5])
ax.set_ylabel('Performance')
ax.set_xlabel('App')
ax.set_title('App Performance')
ax.set_ylim([0,0.8])
for i, text in enumerate(kernel.ratio):
    if text < 0.1: break
    ax.text(i, text+0.02, text,ha = "center", color = 'grey', fontsize = 8)
ax.hlines(0.1, 0, kernel.shape[0], color = 'red')
ax.set_ylabel('')
effect_app = kernel['app'].loc[kernel.ratio >= 0.1].tolist()

kernel = df_train[['channel', 'is_attributed']].copy()
kernel = kernel.groupby(['channel', 'is_attributed']).size()
kernel = kernel.reset_index()
kernel.rename(columns = {0:'cnt'}, inplace = True)
kernel = kernel.pivot(index = 'channel', columns = 'is_attributed', values = 'cnt')
kernel.fillna(0, inplace = True)
kernel['sum'] = kernel.sum(axis=1)
kernel['ratio'] = kernel[1].divide(kernel['sum']).round(2)


ax = plt.subplot2grid((3, 4), (2, 0))
height_ratio = [(kernel['ratio']==0).sum(), (kernel['ratio']!=0).sum()]
ax.bar(x = [0, 1], height = height_ratio, color = ['green', 'red'])
ax.set_xticks([0, 1])

for i, height in enumerate(height_ratio):
    ax.text(i, height+2, height, ha = 'center', color = 'grey')
ax.set_ylim([0,max(height_ratio)+10])
ax.set_title('The # Effectitve channel')

effect_channel = kernel.index[kernel['ratio'] != 0]
effect_channel = kernel.loc[kernel.index.isin(effect_channel),:]
effect_channel.sort_values('ratio', ascending = False, inplace = True)
effect_channel = effect_channel.reset_index()
effect_channel.rename(columns = {'index':'channel'}, inplace = True)
ax = plt.subplot2grid((3, 4), (2, 1), colspan = 3)
sns.barplot(x=effect_channel.index, y="ratio", data=effect_channel, label="Total", palette = sns.color_palette("Blues_d", effect_channel.shape[0]), ax = ax)
#ax.barh(effect_channel.index, effect_channel.ratio,align = 'center', color = 'green')
ax.set_xticks(effect_channel.index)
ax.set_xticklabels(effect_channel.channel)
#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlim([0,0.5])
ax.set_ylabel('Performance')
ax.set_xlabel('Channel')
ax.set_title('Channel Performance')
ax.set_ylim([0,1])
for i, text in enumerate(effect_channel.ratio):
    if text < 0.1: break
    ax.text(i, text+0.02, text,ha = "center", color = 'grey', fontsize = 8)
ax.hlines(0.1, 0, effect_channel.shape[0], color = 'red')
ax.set_ylabel('')
plt.subplots_adjust(wspace = 0.4, hspace = 0.5, top = 0.88)
plt.suptitle('Performance Issue', size = 14)
plt.show()

print('Effect Device: ', effect_device)
print('Effect App: ', effect_app)
print('Effect Channel: ', effect_channel['channel'].loc[effect_channel.ratio >= 0.1].tolist())