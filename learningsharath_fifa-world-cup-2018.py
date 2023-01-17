import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
print("All Packages Loaded Successfully")
sample = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')


sample

sample.loc[sample['country_full'] == 'IR Iran']
table_positions = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
table_positions = table_positions.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
table_positions.country_full.replace("^IR Iran*", "Iran", regex=True, inplace=True)
table_positions['weighted_points'] =  table_positions['cur_year_avg_weighted'] + table_positions['two_year_ago_weighted'] + table_positions['three_year_ago_weighted']
table_positions['rank_date'] = pd.to_datetime(table_positions['rank_date'])
table_positions.head()
matches = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])
matches.head()
world_cup_data = pd.read_csv("../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv")
world_cup_data
