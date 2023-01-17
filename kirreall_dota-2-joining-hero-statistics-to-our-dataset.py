import os

import numpy as np

import pandas as pd


PATH_TO_DATA = '../input/'
df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'test_features.csv'), 

                                    index_col='match_id_hash')
df_train_features.shape
df_train_features.head()
import json



heroes_dict = {}
def add_to_heroes_dict(heroes_dict, path, file_name):

    with open(os.path.join(PATH_TO_DATA, file_name), 'r') as f: # opening file in binary(rb) mode    

        for item in f:

            data_json = json.loads(item)

            for i in range(10):

                heroes_dict[data_json['players'][i]['hero_id']] = data_json['players'][i]['hero_name'].replace('npc_dota_hero_','').replace('_',' ')

  

    return heroes_dict
heroes_dict = add_to_heroes_dict(heroes_dict, PATH_TO_DATA, 'train_matches.jsonl')
heroes_dict = add_to_heroes_dict(heroes_dict, PATH_TO_DATA, 'test_matches.jsonl')
heroes_name_dict = {heroes_dict[k]:k for k in heroes_dict.keys()}
df_heroes_stats = pd.read_html('https://dota2.gamepedia.com/Table_of_hero_attributes', match='HERO', header = 0)[0].drop('A', axis =1)
df_heroes_stats['HERO'] = df_heroes_stats['HERO'].apply(lambda s: s.lower())
df_heroes_stats['hero_id'] = df_heroes_stats['HERO'].map(heroes_name_dict)
df_heroes_stats[df_heroes_stats['hero_id'].isna()]
df_heroes_stats = df_heroes_stats.drop('hero_id', axis =1)
df_heroes_id = pd.DataFrame.from_dict(heroes_name_dict, orient='index')

df_heroes_id.columns=['id']
df_heroes_joined = df_heroes_stats.set_index('HERO').join(df_heroes_id,how='outer')
df_heroes_joined[df_heroes_joined['id'].isna()|df_heroes_joined['STR'].isna()]
heroes_right_name_dict = {'antimage':'anti-mage', 'centaur':'centaur warrunner', 'doom bringer':'doom', 'furion':"nature's prophet", 'queenofpain': 'queen of pain', 'treant':'treant protector', 'treant':'treant protector',

                   'vengefulspirit':'vengeful spirit', 'windrunner':'windranger','zuus':'zeus', 'life stealer':'lifestealer', 'magnataur':'magnus','necrolyte':'necrophos', 'nevermore':'shadow fiend', 

                   'obsidian destroyer':'outworld devourer','rattletrap':'clockwerk','shredder':'timbersaw','wisp':'io', 'abyssal underlord':'underlord', 'skeleton king':'wraith king'}
heroes_name_dict = {heroes_right_name_dict[heroes_dict[k]] if heroes_dict[k] in heroes_right_name_dict.keys() else heroes_dict[k] :k for k in heroes_dict.keys()}
df_heroes_id = pd.DataFrame.from_dict(heroes_name_dict, orient='index')

df_heroes_id.columns=['id']
df_heroes_joined = df_heroes_stats.set_index('HERO').join(df_heroes_id,how='outer')
df_heroes_joined[df_heroes_joined['id'].isna()|df_heroes_joined['STR'].isna()]
df_heroes_joined.drop(['grimstroke','mars'], inplace=True)
def join_hero_statistics(df,df_stat):

    hero_columns = [c for c in df.columns if '_hero_' in c]



    df_features_with_stats = df

    for column in hero_columns:

        df_features_with_stats = df_features_with_stats.join(df_stat.set_index('id'), on = column, rsuffix = '_'+column[:2])



    # After joining we have columns: "STR, STR+, STR25, AGI" etc without "_r1". Rename this columns

    df_features_with_stats.columns = [c+'_r1' if c in df_heroes_joined.columns else c for c in df_features_with_stats.columns]



    return df_features_with_stats
df_train_features_with_stats = join_hero_statistics(df_train_features,df_heroes_joined)
df_test_features_with_stats = join_hero_statistics(df_test_features,df_heroes_joined)
df_train_features_with_stats.head()
df_test_features_with_stats.head()
df_train_features_with_stats.shape
df_test_features_with_stats.shape