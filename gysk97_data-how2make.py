import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import json
import torch
from sklearn.decomposition import PCA
cha_data = pd.read_csv('TFT_Challenger_MatchData.csv')

cha_data = cha_data[:10000]
cha_data.info()
cha_data.head()
combination_column = []
for d in cha_data.combination:
  col = list(json.loads(d.replace("'","\"")).keys())
  for c in col:
    if c not in combination_column:
      combination_column.append(c)
combination_dic  = defaultdict(int)
for c in combination_column:
  combination_dic[c] = 0
for i in range(len(cha_data.combination)):
  dic  = copy.deepcopy(combination_dic)
  col = json.loads(cha_data.combination[i].replace("'","\""))
  for k, v in zip(col.keys(), col.values()):
    dic[k] = v
  cha_data.combination[i] = dic
  
for c in combination_column:
  for i in range(len(cha_data)):
    cha_data.loc[i,c] = cha_data.loc[i,'combination'][c]
cha_data.tail()
champion_column = []
for d in raw_data.champion:
  col = list(json.loads(d.replace("'","\"")).keys())
  for c in col:
    if c not in champion_column:
      champion_column.append(c)
champion_dic  = defaultdict(int)
for c in champion_column:
  champion_dic[c] = 0
for i in range(len(raw_data.champion)):
  dic  = copy.deepcopy(champion_dic)
  col = json.loads(cha_data.champion[i].replace("'","\""))
  for k, v in zip(col.keys(), col.values()):
    if v['star'] == 3:
      dic[k] = 3
    if v['star'] == 2:
      dic[k] = 2
    if v['star'] ==1:
      dic[k] = 1  
  raw_data.champion[i] = dic
for c in champion_column:
  for i in range(len(cha_data)):
    cha_data.loc[i,c] = cha_data.loc[i,'champion'][c]
TFT_data = cha_data.drop(columns = ['gameId','gameDuration', 'combination', 'champion'], axis = 1)
TFT_label =(cha_data['Ranked'])

plt.figure(figsize=(12, 12))
sns.heatmap(cha_data.corr().abs())
TFT_data.Ranked= TFT_data.Ranked >= 4
TFT_data[:-3300].to_csv('moral_TFT_train.csv', index = False)
TFT_data[-3300:].drop('Ranked',axis =1).to_csv('moral_TFT_test.csv', index = False)