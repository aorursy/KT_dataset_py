# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import matplotlib.pyplot as plot

import tensorflow as tf

import random

import torch

import torchvision

from sklearn.model_selection import * 

from catboost import *

from keras import *

from keras.layers.convolutional import Conv2D

from keras.layers import *

from tensorflow.nn import *

from keras.callbacks import *

from keras.models import *

from keras.optimizers import *

from keras.preprocessing import image

from sklearn.metrics import accuracy_score

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
random.seed(2)

np.random.seed(2)

random_seed = 2
train = pandas.read_csv('../input/aiacademy2019/dota2_skill_train.csv', index_col = 'id')

test = pandas.read_csv('../input/aiacademy2019/dota2_skill_test.csv', index_col = 'id')



mine_train = pandas.read_csv('../input/dataframes-sber/df_train.csv')

mine_test = pandas.read_csv('../input/dataframes-sber/df_test.csv')



df_train = pandas.read_csv('../input/lool-this-sucks/Usual_train.csv')

df_test = pandas.read_csv('../input/lool-this-sucks/Usual_test.csv')



best = pandas.read_csv('../input/07999/39420-1389985-submission_solo_4500_nodepth (1).csv')
print("df_test shape = {}".format(df_test.shape[1]))

print("df_train shape = {}".format(df_train.shape[1]))
# df_train = df_train.drop('Unnamed: 0', axis = 1)

# df_test = df_test.drop('Unnamed: 0', axis = 1)
col = list()



for i in df_train:

    col.append(i)

    

columns = pandas.DataFrame({'name': col}, index = None)

columns
# idr = list('')

# i = 0



# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in fin:

#         record = json.loads(line)

#         idr.append(record['id'])

        

# ide = list('')

# i = 0

        

# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in fin:

#         record = json.loads(line)

#         ide.append(record['id'])
# df_train['id'] = train.index

# df_test['id'] = test.index
with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

    for line in fin:

        record = json.loads(line)

        break
# series = record['series']

# plot(series['time'], series['radiant_gold'])

# plot(series['time'], series['dire_gold'])

# plot(series['time'], series['player_gold'])

# title('Gold time series')

# _ = legend(['radiant', 'dire', 'player'])
df_heroes = pandas.read_csv('../input/aiacademy2019/dota2_heroes.csv', index_col='hero_id')
df_heroes.head(2)
# selected_column = 'avg_gpm_x16'

# threshold = 530



# figure(figsize=(12, 4))

# df_train.loc[df_train.skilled == 1, selected_column].hist(bins=100, alpha=0.5)

# df_train.loc[df_train.skilled == 0, selected_column].hist(bins=100, alpha=0.5)

# legend(['skilled', 'not skilled'])

# plot([threshold, threshold], [0, 1000])

# _ = title('Histogram of {}'.format(selected_column))
# skilled_train_pred = (df_train[selected_column] > threshold).astype(int)

# skilled_test_pred = (df_test[selected_column] > threshold).astype(int)
# print('Train accuracy:', (skilled_train_pred == df_train['skilled']).mean())
# df_abilities = pandas.read_csv('../input/aiacademy2019/dota2_abilities.csv', index_col='ability_id')

# df_abilities.behavior = df_abilities.behavior.apply(

#     lambda x: x.split("'")).apply(lambda x: x[1] if len(x) > 1 else 'empty')

# df_abilities.head()
# import tqdm



# l = 0

# ll = 0



# for attack in set(df_heroes.attack_type.fillna('no')):

#     df_train['attack_type_dire_{}'.format(attack)] = 0

#     df_train['attack_type_radiant_{}'.format(attack)] = 0

#     df_test['attack_type_dire_{}'.format(attack)] = 0

#     df_test['attack_type_radiant_{}'.format(attack)] = 0



# for ability_behavior in set(df_abilities.behavior.fillna('empty')):

#     df_train['upgrade_behavior {}'.format(ability_behavior)] = 0

#     df_test['upgrade_behavior {}'.format(ability_behavior)] = 0

    

# for lvl in df_train['level']:

#     df_train['{}_lvl'.format(lvl)] = 0

    

# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for ability_upgrade in record['ability_upgrades']:

#             df_train.loc[record['id'], 'upgrade_behavior {}'.format(

#                 df_abilities.loc[ability_upgrade, 'behavior'])] += 1

            

#         for ll in record['level_up_times']:

#             df_train.loc[record['id'], '{}_lvl'.format(l)] = ll

#             l = l + 1

#         l = 0

        

#         for attack1 in record['dire_heroes']:

#             df_train.loc[record['id'], 'attack_type_dire_{}'.format(df_heroes.loc[attack1, 'attack_type'])] += 1

#         for attack2 in record['radiant_heroes']:

#             df_train.loc[record['id'], 'attack_type_radiant_{}'.format(df_heroes.loc[attack2, 'attack_type'])] += 1



            

            

# for lvl in df_test['level']:

#     df_test['{}_lvl'.format(lvl)] = 0

            

# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for ability_upgrade in record['ability_upgrades']:

#             df_test.loc[record['id'], 'upgrade_behavior {}'.format(

#                 df_abilities.loc[ability_upgrade, 'behavior'])] += 1

            

#         for ll in record['level_up_times']:

#             df_test.loc[record['id'], '{}_lvl'.format(l)] = ll

#             l = l + 1

#         l = 0

        

#         for attack1 in record['dire_heroes']:

#             df_test.loc[record['id'], 'attack_type_dire_{}'.format(df_heroes.loc[attack1, 'attack_type'])] += 1

#         for attack2 in record['radiant_heroes']:

#             df_test.loc[record['id'], 'attack_type_radiant_{}'.format(df_heroes.loc[attack2, 'attack_type'])] += 1
# df_train = df_train.reset_index(drop = True)

# df_test = df_test.reset_index(drop = True)



# df_train = df_train.drop(['25_lvl', '24_lvl'], axis = 1)

# df_test = df_test.drop(['25_lvl', '24_lvl'], axis = 1)



# df_train.to_csv("df_train.csv")

# df_test.to_csv("df_test.csv")
# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for role in eval(df_heroes.loc[record['id'], 'roles']):

#             df_train.loc[record['id'], role] += 1
# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for role in eval(df_heroes.loc[record['id'], 'roles']):

#             df_test.loc[record['id'], role] += 1
# # !pip install gpustat



# try:

#     import simplejson as json

# except ImportError:

#     import json



# import tqdm



# items_len = list('')

# player_gold = list('')

# prebuy = list('')

# all_items = list('')

    

# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         items_len.append(len(record['item_purchase_log'])) 

#         player_gold.append(max(record['series']['player_gold'])) 

#         prebuy.append(sum([i['item_id'] for i in record['item_purchase_log'] if i['time'] < 0])) 

#         all_items.append(sum([d['item_id'] for d in record['item_purchase_log']])) 



# df_train['items_len'] = items_len

# df_train['player_gold'] = player_gold

# df_train['prebuy'] = prebuy

# df_train['all_items'] = all_items



# items_len = list('')

# player_gold = list('')

# prebuy = list('')

# all_items = list('')

    

# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         items_len.append(len(record['item_purchase_log'])) 

#         player_gold.append(max(record['series']['player_gold'])) 

#         prebuy.append(sum([i['item_id'] for i in record['item_purchase_log'] if i['time'] < 0])) 

#         all_items.append(sum([d['item_id'] for d in record['item_purchase_log']])) 

        

# df_test['items_len'] = items_len

# df_test['player_gold'] = player_gold

# df_test['prebuy'] = prebuy

# df_test['all_items'] = all_items
# from sklearn.preprocessing import MultiLabelBinarizer

# mlb = MultiLabelBinarizer()



# items_log = list('')



# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         items_log.append(i['item_id'] for i in record['item_purchase_log'])



# df_train['items_log'] = items_log



# df_train = df_train.join(pandas.DataFrame(mlb.fit_transform(df_train.pop('items_log')), columns = mlb.classes_, index = df_train.index))



# items_log = list('')



# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         items_log.append(i['item_id'] for i in record['item_purchase_log'])



# df_test['items_log'] = items_log



# df_test = df_test.join(pandas.DataFrame(mlb.fit_transform(df_test.pop('items_log')), columns = mlb.classes_, index = df_test.index))
# for hero_id in set(df_train):

#     df_train['dire_hero_{}'.format(hero_id)] = 0

#     df_train['radiant_hero_{}'.format(hero_id)] = 0

    

# for hero_id in set(df_test):

#     df_test['dire_hero_{}'.format(hero_id)] = 0

#     df_test['radiant_hero_{}'.format(hero_id)] = 0
# import tqdm



# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for i in record['dire_heroes']:

#             df_train.loc[record['id'], 'dire_hero_{}'.format(i)] =+ 1

        

# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for i in record['radiant_heroes']:

#             df_train.loc[record['id'], 'radiant_hero_{}'.format(i)] =+ 1

        

# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for i in record['dire_heroes']:

#             df_test.loc[record['id'], 'dire_hero_{}'.format(i)] =+ 1

        

# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in tqdm.tqdm_notebook(fin):

#         record = json.loads(line)

#         for i in record['radiant_heroes']:

#             df_test.loc[record['id'], 'radiant_hero_{}'.format(i)] =+ 1
# df_train.to_csv("df_train.csv")

# df_test.to_csv("df_test.csv")
# display(df_heroes.head(3))

display(df_train.head(3))
from catboost import *

from sklearn.model_selection import *



# for hero_id in set(df_train.hero_id):

#     df_train['is_hero_{}'.format(hero_id)] = df_train.hero_id == hero_id

#     df_test['is_hero_{}'.format(hero_id)] = df_test.hero_id == hero_id



X = df_train

X_test = df_test 

y = mine_train['skilled']



# drop 'player_team', 'winner_team'
print("X_test shape = {}".format(X_test.shape[1]))

print("X shape = {}".format(X.shape[1]))
# X.to_csv("X.csv")

# y.to_csv("y.csv")

# X_test.to_csv("X_test.csv")



# X = pandas.read_csv("../input/x_y_test/X.csv")

# X_test = pandas.read_csv("../input/x_y_test/X_test.csv")

# y = pandas.read_csv("../input/y-data/y.csv")

# y = y.drop("id", axis = 1)
# score_js = ""

# score_js = list(score_js)

# items = ""

# items = list(items)

# ability = "" 

# ability = list(ability)

# max_level_up = "" 

# max_level_up = list(max_level_up)

# max_gold = "" 

# max_gold = list(max_gold)

# max_gold_radiant = "" 

# max_gold_radiant = list(max_gold_radiant)

# max_gold_dire = "" 

# max_gold_dire = list(max_gold_dire)

# play_time = ""

# play_time = list(play_time)

# warmup_time = ""

# warmup_time = list(warmup_time)

# level_up = "" 

# level_up = list(level_up)

# damaged_len = ""

# damaged_len = list(damaged_len)



# with open('../input/aiacademy2019/dota2_skill_train.jsonlines') as fin:

#     for line in fin:

#         record = json.loads(line)

#         score_js.append(record['fight_score'])

#         items.append(sum(record['final_items']))

#         max_level_up.append(max(record['level_up_times']))

#         max_gold.append(max(record['series']['player_gold']))

#         max_gold_radiant.append(max(record['series']['radiant_gold']))

#         max_gold_dire.append(max(record['series']['dire_gold']))

#         play_time.append(abs(max(record['series']['time'])))

#         warmup_time.append(min(record['series']['time']))

#         level_up.append(sum(record['level_up_times']))

#         damaged_len.append(len(record['damage_targets']))

        

# X['fight_score'] = score_js

# X['final_items'] = items

# X['max_level'] = max_level_up

# X['max_player_gold'] = max_gold

# X['max_radiant_gold'] = max_gold_radiant

# X['max_dire_gold'] = max_gold_dire

# X['play_time'] = play_time

# X['warmup_time'] = warmup_time

# X['level_sum'] = level_up

# X['damaged_len'] = damaged_len



# score_js = ""

# score_js = list(score_js)

# items = ""

# items = list(items)

# ability = "" 

# ability = list(ability)

# max_level_up = "" 

# max_level_up = list(max_level_up)

# max_gold = "" 

# max_gold = list(max_gold)

# max_gold_radiant = "" 

# max_gold_radiant = list(max_gold_radiant)

# max_gold_dire = "" 

# max_gold_dire = list(max_gold_dire)

# play_time = ""

# play_time = list(play_time)

# warmup_time = ""

# warmup_time = list(warmup_time)

# level_up = "" 

# level_up = list(level_up)

# damaged_len = ""

# damaged_len = list(damaged_len)



# with open('../input/aiacademy2019/dota2_skill_test.jsonlines') as fin:

#     for line in fin:

#         record = json.loads(line)

#         score_js.append(record['fight_score'])

#         items.append(sum(record['final_items']))

#         max_level_up.append(max(record['level_up_times']))

#         max_gold.append(max(record['series']['player_gold']))

#         max_gold_radiant.append(max(record['series']['radiant_gold']))

#         max_gold_dire.append(max(record['series']['dire_gold']))

#         play_time.append(abs(max(record['series']['time'])))

#         warmup_time.append(min(record['series']['time']))

#         level_up.append(sum(record['level_up_times']))

#         damaged_len.append(len(record['damage_targets']))

        

# X_test['fight_score'] = score_js

# X_test['final_items'] = items

# X_test['max_level'] = max_level_up

# X_test['max_player_gold'] = max_gold

# X_test['max_radiant_gold'] = max_gold_radiant

# X_test['max_dire_gold'] = max_gold_dire

# X_test['play_time'] = play_time

# X_test['warmup_time'] = warmup_time

# X_test['level_sum'] = level_up

# X_test['damaged_len'] = damaged_len

# i = 0

# ii = 0

# 

# for i in range(len(level_up)):

#     for ii in range(len(level_up[i])):

#         X_test['level_is_{}'.format([i])] = X_test.level_up_times[i][ii]
############################################################################################################ X_test



X_test["all_kills"] = X_test["kills"] + X_test["denies"]



X_test["all_gpm_x16"] = X_test["avg_gpm_x16"] + X_test["best_gpm_x16"]

X_test["dif_gpm_x16"] = X_test["best_gpm_x16"] - X_test["avg_gpm_x16"]



X_test["all_xpm_x16"] = X_test["avg_xpm_x16"] + X_test["best_xpm_x16"]

X_test["dif_xpm_x16"] = X_test["best_xpm_x16"] - X_test["avg_xpm_x16"]



X_test["score"] = X_test["fight_score"] + X_test["farm_score"] + X_test["support_score"] + X_test["push_score"]



X_test["avg_KD"] = X_test["avg_kills_x16"] / X_test["avg_deaths_x16"]

X_test["avg_KDA"] = (X_test["avg_kills_x16"] + X_test["avg_assists_x16"]) / X_test["avg_deaths_x16"]

X_test["avg_KDA_r"] = (X_test["avg_kills_x16"] + X_test["avg_assists_x16"] / 2) / X_test["avg_deaths_x16"]



X_test["KD"] = X_test["kills"] / X_test["avg_deaths_x16"]

X_test["KDA"] = (X_test["kills"] + X_test["avg_assists_x16"]) / X_test["avg_deaths_x16"]

X_test["KDA_r"] = (X_test["kills"] + X_test["avg_assists_x16"] / 2) / X_test["avg_deaths_x16"]



X_test["all_gold"] = X_test["gold"] + X_test["gold_spent"]

X_test['all_gold_teams'] = X_test['dire_gold_all'] + X_test['radiant_gold_all']

# X_test["max_gold_diff"] = X_test["max_radiant_gold"] - X_test["max_dire_gold"]



X_test['status_diff'] = X_test['radiant_tower_status'] - X_test['dire_tower_status']



X_test["net_diff"] = X_test["gold_spent"] - X_test["net_worth"]



X_test['not_denies'] = X_test['last_hits'] - X_test['denies']



# X_test['first_blood'] = X_test['first_blood_claimed'] * X_test['first_blood_time']

# X_test = X_test.drop(['first_blood_claimed', 'first_blood_time'], axis = 1)



X_test['per_minute'] = X_test['gold_per_min'] + X_test['xp_per_min']



X_test["full_game"] = X_test["duration"] + X_test["pre_game_duration"]

X_test["full_game_warm"] = X_test["duration"] + X_test["pre_game_duration"] + X_test['warmup_time']

X_test["game_warm"] = X_test["duration"] + X_test['warmup_time']



X_test['dire_percent_gold'] = X_test['dire_gold_all'] / X_test['gold']

X_test['radiant_percent_gold'] = X_test['radiant_gold_all'] / X_test['gold']

X_test['all_gold_percent'] = X_test['all_gold_teams'] / X_test['gold']



############################################################################################################ X



X["all_kills"] = X["kills"] + X["denies"]



X["all_gpm_x16"] = X["avg_gpm_x16"] + X["best_gpm_x16"]

X["dif_gpm_x16"] = X["best_gpm_x16"] - X["avg_gpm_x16"]



X["all_xpm_x16"] = X["avg_xpm_x16"] + X["best_xpm_x16"]

X["dif_xpm_x16"] = X["best_xpm_x16"] - X["avg_xpm_x16"]



X["score"] = X["fight_score"] + X["farm_score"] + X["support_score"] + X["push_score"]



X["avg_KD"] = X["avg_kills_x16"] / X["avg_deaths_x16"]

X["avg_KDA"] = (X["avg_kills_x16"] + X["avg_assists_x16"]) / X["avg_deaths_x16"]

X["avg_KDA_r"] = (X["avg_kills_x16"] + X["avg_assists_x16"] / 2) / X["avg_deaths_x16"]



X["KD"] = X["kills"] / X["avg_deaths_x16"]

X["KDA"] = (X["kills"] + X["avg_assists_x16"]) / X["avg_deaths_x16"]

X["KDA_r"] = (X["kills"] + X["avg_assists_x16"] / 2) / X["avg_deaths_x16"]



X["all_gold"] = X["gold"] + X["gold_spent"]

X['all_gold_teams'] = X['dire_gold_all'] + X['radiant_gold_all']

# X["max_gold_diff"] = X["max_radiant_gold"] - X["max_dire_gold"]



X['status_diff'] = X['radiant_tower_status'] - X['dire_tower_status']



X["net_diff"] = X["gold_spent"] - X["net_worth"]



X['not_denies'] = X['last_hits'] - X['denies']



# X['first_blood'] = X['first_blood_claimed'] * X['first_blood_time']

# X = X.drop(['first_blood_claimed', 'first_blood_time'], axis = 1)



X['per_minute'] = X['gold_per_min'] + X['xp_per_min']



X["full_game"] = X["duration"] + X["pre_game_duration"]

X["full_game_warm"] = X["duration"] + X["pre_game_duration"] + X['warmup_time']

X["game_warm"] = X["duration"] + X['warmup_time']



X['dire_percent_gold'] = X['dire_gold_all'] / X['gold']

X['radiant_percent_gold'] = X['radiant_gold_all'] / X['gold']

X['all_gold_percent'] = X['all_gold_teams'] / X['gold']
print("Y rows = {}".format(y.shape[0]))

print("X rows = {}".format(X.shape[0])) 
y = y.head(len(X))
y
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .3, random_state = 1)
# cat_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
from sklearn.ensemble import *

from catboost import *

from sklearn.model_selection import *

from sklearn.metrics import *



model_cb = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 10000, random_seed = 512, task_type = 'GPU', verbose = 0, learning_rate = 0.04)    

model_cb.fit(X, y)                                    # eval_set = (X_val, y_val), use_best_model = True, plot = True) # (cat_features = cat_features)

y_pred_cb = model_cb.predict_proba(X_test)[:, 1]

y_pred = (y_pred_cb > 0.45).astype(int)

df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

df_submission.to_csv('submission_solo_23500_nodepth_512.csv') 



# df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

# submis = (df_submission.skilled * 0.46 + best.skilled * 0.54)

# submis = (submis > 0.45).astype(int)

# subm = pandas.DataFrame({'skilled': submis}, index = test.index)

# subm.to_csv('merged.csv') # submission_solo_4500_nodepth_512

# subm.tail()



# params = {'random_seed': [512], 'n_estimators': [4200, 4250, 4300, 4350, 4400]}

# cb = CatBoostClassifier(eval_metric = "Accuracy", task_type = 'GPU', verbose = 0, learning_rate = 0.05)    

# model_grid_cb = GridSearchCV(cb, params, scoring = 'accuracy', cv = 3, verbose = 3, n_jobs = -1)

# model_grid_cb.fit(X, y)



# print(params.get('n_estimators')) 



'''



model_cb = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 1200, random_seed = 70, task_type = 'GPU', verbose = 0)

model_cb.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cb = model_cb.predict_proba(X_test)[:, 1]



model_cbo = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 2400, random_seed = 70, task_type = 'GPU', verbose = 0)

model_cbo.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cbo = model_cbo.predict_proba(X_test)[:, 1]



y_pred = ((0.4 * y_pred_cb + 0.6 * y_pred_cbo) > 0.45).astype(int)

df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

df_submission.to_csv('submission_ensemble2.csv')



model_cb = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 3300, random_seed = 70, learning_rate = 0.05, task_type = 'GPU', verbose = 0)

model_cb.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cb = model_cb.predict_proba(X_test)[:, 1]



model_cbo = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 6800, random_seed = 70, learning_rate = 0.05, task_type = 'GPU', verbose = 0)

model_cbo.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cbo = model_cbo.predict_proba(X_test)[:, 1]



model_cba = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 4700, random_seed = 70, learning_rate = 0.05, task_type = 'GPU', verbose = 0)

model_cba.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cba = model_cba.predict_proba(X_test)[:, 1]



y_pred = ((0.2 * y_pred_cb + 0.5 * y_pred_cbo + 0.3 * y_pred_cba) > 0.45).astype(int)

df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

df_submission.to_csv('submission_ensemble3.csv')

df_submission.tail()



'''
# y_pred_cb = model_grid_cb.predict_proba(X_test)[:, 1]

# y_pred = (y_pred_cb > 0.45).astype(int)

# df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

# df_submission.to_csv('submission_solo_4500_nodepth_512.csv')

# df_submission.tail()
# model_grid_cb.best_params_
# display(df_submission)
'''



import lightgbm as lgb



params = {'application': 'binary',

          'boosting': 'gbdt',

          'metric': 'auc',

          'num_leaves': 70,

          'max_depth': 9,

          'learning_rate': 0.01,

          'bagging_fraction': 0.85,

          'feature_fraction': 0.8,

          'min_split_gain': 0.02,

          'min_child_samples': 150,

          'min_child_weight': 0.02,

          'lambda_l2': 0.0475,

          'verbosity': -1,

          'data_random_seed': 17,

          'verbosity': 0,

          'boost_from_average': False}



params2 = {'application': 'binary',

          'boosting': 'gbdt',

          'metric': 'auc',

          'num_leaves': 60,

          'max_depth': 8,

          'learning_rate': 0.05,

          'bagging_fraction': 0.85,

          'feature_fraction': 0.8,

          'min_split_gain': 0.02,

          'min_child_samples': 150,

          'min_child_weight': 0.02,

          'lambda_l2': 0.0475,

          'verbosity': -1,

          'data_random_seed': 1337,

          'verbosity': 0,

          'boost_from_average': False}



# Additional parameters:

early_stop = 100

verbose_eval = 100

num_rounds = 2200 # 2500

n_splits = 7

early_stop3 = 300

verbose_eval3 = 100

num_rounds3 = 2



'''
'''



from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostRegressor, CatBoostClassifier

from mlxtend.regressor import StackingRegressor



kfold = StratifiedKFold(n_splits = n_splits, random_state = 1337)



d_train = lgb.Dataset(X_train, label = y_train)

d_valid = lgb.Dataset(X_val, label = y_val)

watchlist = [d_train, d_valid]



print('training:')



model2 = lgb.train(params, 

                  train_set = d_train,

                  num_boost_round = num_rounds,

                  valid_sets = watchlist,

                  verbose_eval = verbose_eval,

                  early_stopping_rounds = early_stop)



model4 = lgb.train(params2,

                  train_set = d_train,

                  num_boost_round = num_rounds,

                  valid_sets = watchlist,

                  verbose_eval = verbose_eval,

                  early_stopping_rounds = early_stop)



model_cb = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 17300, random_seed = 70, learning_rate = 0.01, task_type = 'GPU', verbose = 0)

model_cb.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cb = model_cb.predict_proba(X_test)[:, 1]



model_cbo = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 35000, random_seed = 70, learning_rate = 0.01, task_type = 'GPU', verbose = 0)

model_cbo.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cbo = model_cbo.predict_proba(X_test)[:, 1]



model_cba = CatBoostClassifier(eval_metric = "Accuracy", n_estimators = 24500, random_seed = 70, learning_rate = 0.01, task_type = 'GPU', verbose = 0)

model_cba.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True)

y_pred_cba = model_cba.predict_proba(X_test)[:, 1]



y_pred_cbc = ((0.2 * y_pred_cb + 0.5 * y_pred_cbo + 0.3 * y_pred_cba) > 0.45).astype(int)

y_pred_lgba = model2.predict(X_test)

y_pred_lgbb = model4.predict(X_test)



y_pred = ((0.4 * y_pred_lgba + 0.3 * y_pred_lgbb + 0.3 * y_pred_cbc) > 0.45).astype(int)

df_submission = pandas.DataFrame({'skilled': y_pred}, index = test.index)

df_submission.to_csv('submission_ensemble_lg_c.csv')

df_submission.tail()



'''
# model_cb = CatBoostClassifier(n_estimators = 1, depth = 1)

# model_cb.fit(X_train, y_train)
# df_submission = pandas.DataFrame({'skilled': y_pred.astype(int)}, index = df_test.index)



# df_submission.to_csv('submission.csv')
# df_submission = pandas.DataFrame({'skilled': y_pred_inv.astype(int)}, index = df_test.index)
# d = list('')



# for i in range(43265):

#     x = randint(0, 2)

#     d.append(x)

#     df_submission.loc[-1] = d  # adding a row

#     df_submission.index = df_submission.index + 1  # shifting index

#     df_submission = df_submission.sort_index()  # sorting by index

#     d = list('')

#     print(i)
# df_submission['skilled'] = d



# df_submission.to_csv('submission_ensemble_rand.csv')

# df_submission.tail()



# df_submission['skilled'] = df_submission['skilled'].map({0: 1, 1: 0})



# df_submission.to_csv('submission_ensemble_rand_inv.csv')

# df_submission.tail()
feature_score = pandas.DataFrame(list(zip(X.dtypes.index, model_cb.get_feature_importance(Pool(X, label = y)))), columns = ['Feature','Score']) 

feature_score = feature_score.sort_values(by = 'Score', ascending = False, inplace = False, kind = 'quicksort', na_position = 'last')



feature_score = feature_score.reset_index(drop = True)
feature_score.to_csv("importance.csv", index = False)
feature_score.tail(60)