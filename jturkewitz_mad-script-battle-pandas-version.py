# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:52:25 2016

@author: Jared Turkewitz
"""
#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import timeit

tic0=timeit.default_timer()
tic=timeit.default_timer()
pd.options.mode.chained_assignment = None  # default='warn'

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

toc=timeit.default_timer()
print('Load Time',toc - tic)

#%%
def get_grid_dict(x_bins=100,y_bins=100):

    grid_dict = {}
    grid_value = 0
    for i in range(x_bins):
        for j in range(y_bins):
            key = (i,j)
            grid_dict[key] = grid_value
            grid_value = grid_value + 1
    return grid_dict
#%%
tic=timeit.default_timer()
tic1=timeit.default_timer()


y_bins_fine = []
y_bins_range = 800
for i in range(y_bins_range):
    y_bins_fine.append(i/(y_bins_range-1)*10  )
x_bins_fine = []
x_bins_range = 400
for i in range(x_bins_range):
    x_bins_fine.append(i/(x_bins_range-1)*10 )

#all caps to avoid spyder variable explorer bug
GRID_DICT_FINE = get_grid_dict(x_bins_range,y_bins_range)

#normally I combine train and test before feature engineering but this causes memory issues
#for the scripts with the zip step
train['y_binned_fine'] = np.digitize(train['y'],y_bins_fine,right=True)
train['x_binned_fine'] = np.digitize(train['x'],x_bins_fine,right=True)
train['xy_binned_fine'] = list(zip(train['x_binned_fine'],train['y_binned_fine']))
train['xy_binned_fine'] = train['xy_binned_fine'].map(GRID_DICT_FINE)

test['y_binned_fine'] = np.digitize(test['y'],y_bins_fine,right=True)
test['x_binned_fine'] = np.digitize(test['x'],x_bins_fine,right=True)
test['xy_binned_fine'] = list(zip(test['x_binned_fine'],test['y_binned_fine']))
test['xy_binned_fine'] = test['xy_binned_fine'].map(GRID_DICT_FINE)

toc=timeit.default_timer()
print('Getting xy time',toc - tic)

tic=timeit.default_timer()

train_grid_fine_grouped = train.groupby('xy_binned_fine')
temp_vc = train_grid_fine_grouped['place_id'].value_counts()

temp_df = pd.DataFrame(temp_vc)
temp_df.rename(columns={'place_id':'freq'},inplace=True)
temp_df.reset_index(inplace=True)

temp_df_grouped_place_id = temp_df.groupby('xy_binned_fine')['place_id']

#get 3 most common places in the given grid xy bins
COMMON_1_DICT = temp_df_grouped_place_id.nth(0).to_dict()
COMMON_2_DICT = temp_df_grouped_place_id.nth(1).to_dict()
COMMON_3_DICT = temp_df_grouped_place_id.nth(2).to_dict()

def apply_common_dict(common_dict,x):
    try:
        return common_dict[x]
    except KeyError:
        return -1

test['pred_0'] = test['xy_binned_fine'].map(lambda x: apply_common_dict(COMMON_1_DICT,x))
test['pred_1'] = test['xy_binned_fine'].map(lambda x: apply_common_dict(COMMON_2_DICT,x))
test['pred_2'] = test['xy_binned_fine'].map(lambda x: apply_common_dict(COMMON_3_DICT,x))

toc=timeit.default_timer()
print('Grouping Time',toc - tic)

#if you want to do cv
#test['pred_0_res'] = test['pred_0'] - test['place_id']
#test['pred_1_res'] = test['pred_1'] - test['place_id']
#test['pred_2_res'] = test['pred_2'] - test['place_id']
#test['pred_0_res'] = (test['pred_0_res'] == 0).astype(int)
#test['pred_1_res'] = (test['pred_1_res'] == 0).astype(int) / 2
#test['pred_2_res'] = (test['pred_2_res'] == 0).astype(int) / 3
#test['apk'] = test['pred_0_res'] + test['pred_1_res'] + test['pred_2_res']
#print('apk',test['apk'].mean())

toc=timeit.default_timer()
print('Running Time',toc - tic1)
#%%

tic=timeit.default_timer()

test_neg0_cond = test['pred_0'] == -1
test_neg1_cond = test['pred_1'] == -1
test_neg2_cond = test['pred_2'] == -1

test['pred_0'][test_neg0_cond] = 8772469670
test['pred_1'][test_neg1_cond] = 1623394281
test['pred_2'][test_neg2_cond] = 1308450003

test['pred_0'] = test['pred_0'].astype(str)
test['pred_1'] = test['pred_1'].astype(str)
test['pred_2'] = test['pred_2'].astype(str)

test['place_id'] = test['pred_0'] + ' ' + test['pred_1'] + ' ' + test['pred_2']

test = test[['row_id','place_id']].copy()
test.to_csv('basic_fb_checkin.csv', index=False)
print('submission created')
toc=timeit.default_timer()
print('Making Sub Time',toc - tic)
toc=timeit.default_timer()
print('Total Time',toc - tic0)

COMMON_1_DICT[0]
test['y_binned_fine'] = np.digitize(test['y'],y_bins_fine,right=True)
test['x_binned_fine'] = np.digitize(test['x'],x_bins_fine,right=True)
test['xy_binned_fine'] = list(zip(test['x_binned_fine'],test['y_binned_fine']))
test['xy_binned_fine'] = test['xy_binned_fine'].map(GRID_DICT_FINE)
test = pd.read_csv("../input/test.csv")
test['y_binned_fine'] = np.digitize(test['y'],y_bins_fine,right=True)
test['x_binned_fine'] = np.digitize(test['x'],x_bins_fine,right=True)
test['xy_binned_fine'] = list(zip(test['x_binned_fine'],test['y_binned_fine']))
test['xy_binned_fine'] = test['xy_binned_fine'].map(GRID_DICT_FINE)
test['pred_0'] = test['xy_binned_fine'].map(lambda x: apply_common_dict(COMMON_1_DICT,x))
test['pred_0']
y_bins_fine
test['xy_binned_fine'] = list(zip(test['x_binned_fine'],test['y_binned_fine']))
7*7


