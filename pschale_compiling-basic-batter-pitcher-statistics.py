import pandas as pd

from time import time

import matplotlib.pyplot as plt

import numpy as np
t = time()

atbats = pd.read_csv('../input/atbats.csv', index_col=0)

games = pd.read_csv('../input/games.csv', index_col=0)

print(time()-t)
enc_dict = {'Single': '1B','Double': '2B','Triple': '3B','Home Run': 'HR','Walk': 'BB',

            'Intent Walk': 'IBB','Hit By Pitch': 'HBP','Strikeout': 'K','Sac Fly': 'SF',

            'Grounded Into DP': 'GIDP','Groundout': 'GO','Lineout': 'LO','Pop Out': 'PO',

            'Flyout': 'FO','Fielders Choice': 'FC','Sac Bunt': 'SAC','Double Play': 'DP',

            'Triple Play': 'TP','Batter Interference': 'BI','Fan interference': 'FI',

            'Catcher Interference': 'CI','Field Error': 'ROE','Bunt Groundout': 'BGO',

            'Bunt Lineout': 'BLO','Bunt Pop Out': 'BPO','Fielders Choice Out': 'FCO',

            'Forceout': 'FORCE','Sacrifice Bunt DP': 'SBDP','Strikeout - DP': 'KDP',

            'Runner Out': 'RO','Sac Fly DP': 'SFDP'

           }
t = time()

atbats['year'] = (atbats.index//1e6).astype(int)

atbats['event'] = atbats['event'].apply(lambda x: enc_dict[x])

eventcol = atbats['event']

atbats = pd.get_dummies(atbats, columns=['event'], prefix='')

atbats['event'] = eventcol

print(time()-t)
atbats['_H'] = atbats[['_1B', '_2B', '_3B', '_HR']].sum(axis=1)

atbats['_TB'] = atbats['_1B'] + 2*atbats['_2B'] + 3*atbats['_3B'] + 4*atbats['_HR']

atbats['_K'] = atbats['_K'] + atbats['_KDP']

atbats['_BB'] = atbats['_BB'] + atbats['_IBB']

atbats['_AB'] = 1 - atbats[['_CI', '_SAC', '_BB', 

                            '_HBP', '_RO', '_SF']].sum(axis=1)

atbats['_PA'] = 1 - atbats['_RO']
atbats['_outs'] = atbats['o'] - atbats.groupby(['g_id', 

                                                'inning', 

                                                'top'])['o'].shift(1).fillna(0)
stats_to_cum = pd.Index(['H', 'TB', 'PA', 'AB', 'BB', 'HBP', 'IBB', 'K', 

                         '2B', '3B', 'HR', 'GIDP', 'SF'])



batter_groups = atbats.groupby(['batter_id', 'year'])

pitcher_groups = atbats.groupby(['pitcher_id', 'year'])



atbats[stats_to_cum] = (batter_groups['_' + stats_to_cum]

                                     .transform(pd.Series.cumsum)).astype(int)



atbats['opp_' + stats_to_cum] = (pitcher_groups['_' + stats_to_cum]

                                               .transform(pd.Series.cumsum)).astype(int)



atbats['IP'] = pitcher_groups['_outs'].transform(pd.Series.cumsum) / 3
atbats['AVG'] = atbats['H']/atbats['AB']

atbats['SLG'] = atbats['TB']/atbats['AB']

atbats['OBP'] = atbats[['H', 'BB', 'HBP']].sum(axis=1)/atbats[['AB', 'BB', 'HBP', 

                                                               'SF']].sum(axis=1)

atbats['OPS'] = atbats['SLG'] + atbats['OBP']

atbats['K%'] = atbats['K'] / atbats['AB']

atbats['BB%'] = atbats['BB'] / atbats['AB']

atbats['K-BB%'] = atbats['K%'] - atbats['BB%']

atbats['BABIP'] = (atbats['H'] - atbats['HR'])/(atbats['AB'] + atbats['SF'] 

                                                          - atbats['HR'] - atbats['K'])

atbats['ISO'] = atbats['SLG'] - atbats['AVG']
atbats['opp_AVG'] = atbats['opp_H']/atbats['opp_AB']

atbats['opp_SLG'] = atbats['opp_TB']/atbats['opp_AB']

atbats['opp_OBP'] = (atbats[['opp_H', 'opp_BB', 'opp_HBP']].sum(axis=1) / 

                             atbats[['opp_AB', 'opp_BB', 

                                     'opp_HBP', 'opp_SF']].sum(axis=1))

atbats['opp_OPS'] = atbats['opp_SLG'] + atbats['opp_OBP']

atbats['opp_K%'] = atbats['opp_K'] / atbats['opp_AB']

atbats['opp_BB%'] = atbats['opp_BB'] / atbats['opp_AB']

atbats['opp_K-BB%'] = atbats['opp_K%'] - atbats['opp_BB%']

atbats['opp_BABIP'] = ((atbats['opp_H'] - atbats['opp_HR'])/

                               (atbats['opp_AB'] + atbats['opp_SF'] - 

                                atbats['opp_HR'] - atbats['opp_K']))



atbats['FIP'] = (13*atbats['opp_HR'] + 3*(atbats['opp_BB'] + atbats['opp_HBP']) 

                 - 2.0*atbats['opp_K'])/atbats['IP'] + 3.2

atbats['WHIP'] = atbats[['opp_H', 'opp_BB']].sum(axis=1)/atbats['IP']
player_names = pd.read_csv('../input/player_names.csv', index_col=0)

atbats = atbats.merge(player_names, left_on='batter_id', right_index=True, how='left')

atbats = atbats.merge(player_names, left_on='pitcher_id', right_index=True, 

                      how='left', suffixes=['_batter', '_pitcher'])
(atbats[atbats['AB']>400].groupby(['batter_id', 'year']).tail(n=1)

                                               .sort_values(by='OPS', ascending=False)

                                               .set_index(['year', 'batter_id'])

                                                [['AB', 'H', '2B', '3B', 'HR', 'BB', 

                                                  'IBB', 'K', 'AVG', 'SLG', 'OBP', 'OPS', 

                                                  'first_name_batter', 

                                                  'last_name_batter']]

                                               .head(n=20))
(atbats[atbats['IP']>100].groupby(['pitcher_id', 'year'])

                        .tail(n=1)

                        .sort_values(by='FIP').set_index(['year', 'pitcher_id'])

                         [['WHIP', 'FIP', 'IP', 'opp_K', 'opp_BB',

                           'first_name_pitcher', 'last_name_pitcher']]

                        .head(n=20))