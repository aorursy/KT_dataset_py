import pandas as pd

import pandas_profiling

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as mticker

from math import pi

import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from datetime import datetime

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None
# The two variables below might be changed once a new version of the dataset will be published or once different charts will be required



end_date = '2020-01-31'

smoothing_factor = 2
df16_players = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin16_players.csv")

df17_players = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin17_players.csv")

df18_players = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin18_players.csv")

df19_players = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin19_players.csv")

df20_players = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin20_players.csv")



df16_prices = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin16_prices.csv")

df17_prices = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin17_prices.csv")

df18_prices = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin18_prices.csv")

df19_prices = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin19_prices.csv")

df20_prices = pd.read_csv("../input/fifa-20-ultimate-team-players-dataset/fut_bin20_prices.csv")



df16_prices['date'] = pd.to_datetime(df16_prices['date'])

df17_prices['date'] = pd.to_datetime(df17_prices['date'])

df18_prices['date'] = pd.to_datetime(df18_prices['date'])

df19_prices['date'] = pd.to_datetime(df19_prices['date'])

df20_prices['date'] = pd.to_datetime(df20_prices['date'])

df16_prices[['ps4', 'xbox']] = df16_prices[['ps4', 'xbox']].apply(pd.to_numeric, errors='coerce')

df17_prices[['ps4', 'xbox']] = df17_prices[['ps4', 'xbox']].apply(pd.to_numeric, errors='coerce')

df18_prices[['ps4', 'xbox']] = df18_prices[['ps4', 'xbox']].apply(pd.to_numeric, errors='coerce')

df19_prices[['ps4', 'xbox']] = df19_prices[['ps4', 'xbox']].apply(pd.to_numeric, errors='coerce')

df20_prices[['ps4', 'xbox']] = df20_prices[['ps4', 'xbox']].apply(pd.to_numeric, errors='coerce')
df20_players.head()
df20_prices.head()
resource_ids = {'CR7': 20801,

                'Messi': 158023}



column_names = {'pace': 'Pace', 'dribbling': 'Dribble', 'shooting': 'Shoot',

                'passing': 'Pass', 'defending': 'Defend', 'physicality': 'Physic',

                'pace_acceleration': 'Accel', 'pace_sprint_speed': 'Sprint', 'phys_jumping': 'Jump',

                'phys_stamina': 'Stamina', 'phys_strength': 'Strength', 'phys_aggression': 'Aggress',

                'drib_agility': 'Agility', 'drib_balance': 'Balance', 'drib_reactions': 'React',

                'drib_ball_control': 'Ball ctrl', 'drib_dribbling': 'Dribbl', 'drib_composure': 'Compos',

                'shoot_positioning': 'Posit', 'shoot_finishing': 'Finish', 'shoot_shot_power': 'Shot power',

                'shoot_long_shots': 'Long shot', 'shoot_volleys': 'Volleys', 'shoot_penalties': 'Penalty'}



df16_filtered = df16_players[df16_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df16_filtered.rename(columns=column_names, inplace=True)

df17_filtered = df17_players[df17_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df17_filtered.rename(columns=column_names, inplace=True)

df18_filtered = df18_players[df18_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df18_filtered.rename(columns=column_names, inplace=True)

df19_filtered = df19_players[df19_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df19_filtered.rename(columns=column_names, inplace=True)

df20_filtered = df20_players[df20_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df20_filtered.rename(columns=column_names, inplace=True)



attribute_columns_filter = ['Pace', 'Dribble', 'Shoot', 'Pass', 'Defend', 'Physic']

df16_attributes = df16_filtered[attribute_columns_filter]

df17_attributes = df17_filtered[attribute_columns_filter]

df18_attributes = df18_filtered[attribute_columns_filter]

df19_attributes = df19_filtered[attribute_columns_filter]

df20_attributes = df20_filtered[attribute_columns_filter]



physical_columns_filter = ['Accel', 'Sprint', 'Jump', 'Stamina', 'Strength', 'Aggress']

df16_physical = df16_filtered[physical_columns_filter]

df17_physical = df17_filtered[physical_columns_filter]

df18_physical = df18_filtered[physical_columns_filter]

df19_physical = df19_filtered[physical_columns_filter]

df20_physical = df20_filtered[physical_columns_filter]



dribbling_columns_filter = ['Agility', 'Balance', 'React', 'Ball ctrl', 'Dribbl', 'Compos']

df16_dribbling = df16_filtered[dribbling_columns_filter]

df17_dribbling = df17_filtered[dribbling_columns_filter]

df18_dribbling = df18_filtered[dribbling_columns_filter]

df19_dribbling = df19_filtered[dribbling_columns_filter]

df20_dribbling = df20_filtered[dribbling_columns_filter]



shooting_columns_filter = ['Posit', 'Finish', 'Shot power', 'Long shot', 'Volleys', 'Penalty']

df16_shooting = df16_filtered[shooting_columns_filter]

df17_shooting = df17_filtered[shooting_columns_filter]

df18_shooting = df18_filtered[shooting_columns_filter]

df19_shooting = df19_filtered[shooting_columns_filter]

df20_shooting = df20_filtered[shooting_columns_filter]
plt.figure(figsize=(20, 12))



attributes = list(df16_attributes)

att_no = len(attributes)

values1 = df16_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 1, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.title('FIFA 16', size=11, y=1.1)



attributes = list(df17_attributes)

att_no = len(attributes)

values1 = df17_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 2, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

plt.title('FIFA 17', size=11, y=1.1)



attributes = list(df18_attributes)

att_no = len(attributes)

values1 = df18_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 3, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

plt.title('FIFA 18', size=11, y=1.1)



attributes = list(df19_attributes)

att_no = len(attributes)

values1 = df19_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 4, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

plt.title('FIFA 19', size=11, y=1.1)



attributes = list(df20_attributes)

att_no = len(attributes)

values1 = df20_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 5, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

plt.title('FIFA 20', size=11, y=1.1)



attributes = list(df16_physical)

att_no = len(attributes)

values1 = df16_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 6, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df17_physical)

att_no = len(attributes)

values1 = df17_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 7, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df18_physical)

att_no = len(attributes)

values1 = df18_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 8, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df19_physical)

att_no = len(attributes)

values1 = df19_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 9, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df20_physical)

att_no = len(attributes)

values1 = df20_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 10, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df16_dribbling)

att_no = len(attributes)

values1 = df16_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 11, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df17_dribbling)

att_no = len(attributes)

values1 = df17_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 12, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df18_dribbling)

att_no = len(attributes)

values1 = df18_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 13, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df19_dribbling)

att_no = len(attributes)

values1 = df19_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 14, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df20_dribbling)

att_no = len(attributes)

values1 = df20_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 15, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df16_shooting)

att_no = len(attributes)

values1 = df16_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 16, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df17_shooting)

att_no = len(attributes)

values1 = df17_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 17, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df18_shooting)

att_no = len(attributes)

values1 = df18_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 18, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df19_shooting)

att_no = len(attributes)

values1 = df19_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 19, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)



attributes = list(df20_shooting)

att_no = len(attributes)

values1 = df20_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

ax = plt.subplot(4, 5, 20, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)
df_prices_ps4 = pd.concat([

df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

            (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

            (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

            (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

            (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

            (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'ps4']].set_index('date')])

df_prices_xbox = pd.concat([

    df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

                (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

                (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

                (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

                (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

                (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'xbox']].set_index('date')])

if len(list(resource_ids.keys())) == 2:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4_3 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[2]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2, df_prices_ps4_3], axis=1)

df_prices_ps4.reset_index(inplace=True)

if len(list(resource_ids.keys())) == 2:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox_3 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[2]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2, df_prices_xbox_3], axis=1)

df_prices_xbox.reset_index(inplace=True)
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['CR7'], sigma=smoothing_factor), label='CR7', color='C0')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Messi'], sigma=smoothing_factor), label='Messi', color='C1')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - CR7 vs Messi prices on PS4')

plt.gcf().autofmt_xdate()

plt.show()
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['CR7'], sigma=smoothing_factor), label='CR7', color='C9')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Messi'], sigma=smoothing_factor), label='Messi', color='C8')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - CR7 vs Messi prices on Xbox')

plt.gcf().autofmt_xdate()

plt.show()
df_prices_ps4.columns = ['date', 'CR7 - PS4', 'Messi - PS4']

df_prices_xbox.columns = ['date', 'CR7 - Xbox', 'Messi - Xbox']

df_prices = df_prices_ps4.merge(df_prices_xbox, on='date')

df_prices['CR7'] = ((df_prices['CR7 - PS4'] / df_prices['CR7 - Xbox']) - 1) * 100

df_prices['Messi'] = ((df_prices['Messi - PS4'] / df_prices['Messi - Xbox']) - 1) * 100

df_prices = df_prices[['date', 'CR7', 'Messi']]
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['CR7'], sigma=smoothing_factor), label='CR7', color='C4')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Messi'], sigma=smoothing_factor), label='Messi', color='C2')

ax.set_ylabel('Δ PS4 vs Xbox price (in %)')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - CR7 vs Messi ratio between PS4 and Xbox prices')

plt.gcf().autofmt_xdate()

plt.show()
resource_ids = {'Firmino': 201942,

                'Mane': 208722,

                'Salah': 209331}



column_names = {'pace': 'Pace', 'dribbling': 'Dribble', 'shooting': 'Shoot',

                'passing': 'Pass', 'defending': 'Defend', 'physicality': 'Physic',

                'pace_acceleration': 'Accel', 'pace_sprint_speed': 'Sprint', 'phys_jumping': 'Jump',

                'phys_stamina': 'Stamina', 'phys_strength': 'Strength', 'phys_aggression': 'Aggress',

                'drib_agility': 'Agility', 'drib_balance': 'Balance', 'drib_reactions': 'React',

                'drib_ball_control': 'Ball ctrl', 'drib_dribbling': 'Dribbl', 'drib_composure': 'Compos',

                'shoot_positioning': 'Posit', 'shoot_finishing': 'Finish', 'shoot_shot_power': 'Shot power',

                'shoot_long_shots': 'Long shot', 'shoot_volleys': 'Volleys', 'shoot_penalties': 'Penalty'}



df16_filtered = df16_players[df16_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df16_filtered.rename(columns=column_names, inplace=True)

df17_filtered = df17_players[df17_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df17_filtered.rename(columns=column_names, inplace=True)

df18_filtered = df18_players[df18_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df18_filtered.rename(columns=column_names, inplace=True)

df19_filtered = df19_players[df19_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df19_filtered.rename(columns=column_names, inplace=True)

df20_filtered = df20_players[df20_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df20_filtered.rename(columns=column_names, inplace=True)



attribute_columns_filter = ['Pace', 'Dribble', 'Shoot', 'Pass', 'Defend', 'Physic']

df16_attributes = df16_filtered[attribute_columns_filter]

df17_attributes = df17_filtered[attribute_columns_filter]

df18_attributes = df18_filtered[attribute_columns_filter]

df19_attributes = df19_filtered[attribute_columns_filter]

df20_attributes = df20_filtered[attribute_columns_filter]



physical_columns_filter = ['Accel', 'Sprint', 'Jump', 'Stamina', 'Strength', 'Aggress']

df16_physical = df16_filtered[physical_columns_filter]

df17_physical = df17_filtered[physical_columns_filter]

df18_physical = df18_filtered[physical_columns_filter]

df19_physical = df19_filtered[physical_columns_filter]

df20_physical = df20_filtered[physical_columns_filter]



dribbling_columns_filter = ['Agility', 'Balance', 'React', 'Ball ctrl', 'Dribbl', 'Compos']

df16_dribbling = df16_filtered[dribbling_columns_filter]

df17_dribbling = df17_filtered[dribbling_columns_filter]

df18_dribbling = df18_filtered[dribbling_columns_filter]

df19_dribbling = df19_filtered[dribbling_columns_filter]

df20_dribbling = df20_filtered[dribbling_columns_filter]



shooting_columns_filter = ['Posit', 'Finish', 'Shot power', 'Long shot', 'Volleys', 'Penalty']

df16_shooting = df16_filtered[shooting_columns_filter]

df17_shooting = df17_filtered[shooting_columns_filter]

df18_shooting = df18_filtered[shooting_columns_filter]

df19_shooting = df19_filtered[shooting_columns_filter]

df20_shooting = df20_filtered[shooting_columns_filter]
plt.figure(figsize=(20, 12))



attributes = list(df16_attributes)

att_no = len(attributes)

values1 = df16_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 1, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 16', size=11, y=1.1)



attributes = list(df17_attributes)

att_no = len(attributes)

values1 = df17_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 2, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 17', size=11, y=1.1)



attributes = list(df18_attributes)

att_no = len(attributes)

values1 = df18_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 3, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 18', size=11, y=1.1)



attributes = list(df19_attributes)

att_no = len(attributes)

values1 = df19_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 4, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 19', size=11, y=1.1)



attributes = list(df20_attributes)

att_no = len(attributes)

values1 = df20_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 5, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 20', size=11, y=1.1)



attributes = list(df16_physical)

att_no = len(attributes)

values1 = df16_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 6, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df17_physical)

att_no = len(attributes)

values1 = df17_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 7, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df18_physical)

att_no = len(attributes)

values1 = df18_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 8, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df19_physical)

att_no = len(attributes)

values1 = df19_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 9, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df20_physical)

att_no = len(attributes)

values1 = df20_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 10, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df16_dribbling)

att_no = len(attributes)

values1 = df16_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 11, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df17_dribbling)

att_no = len(attributes)

values1 = df17_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 12, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df18_dribbling)

att_no = len(attributes)

values1 = df18_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 13, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df19_dribbling)

att_no = len(attributes)

values1 = df19_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 14, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df20_dribbling)

att_no = len(attributes)

values1 = df20_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 15, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df16_shooting)

att_no = len(attributes)

values1 = df16_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 16, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df17_shooting)

att_no = len(attributes)

values1 = df17_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 17, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df18_shooting)

att_no = len(attributes)

values1 = df18_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 18, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df19_shooting)

att_no = len(attributes)

values1 = df19_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 19, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df20_shooting)

att_no = len(attributes)

values1 = df20_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 20, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)
df_prices_ps4 = pd.concat([

df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

            (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

            (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

            (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

            (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

            (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'ps4']].set_index('date')])

df_prices_xbox = pd.concat([

    df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

                (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

                (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

                (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

                (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

                (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'xbox']].set_index('date')])

if len(list(resource_ids.keys())) == 2:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4_3 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[2]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2, df_prices_ps4_3], axis=1)

df_prices_ps4.reset_index(inplace=True)

if len(list(resource_ids.keys())) == 2:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox_3 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[2]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2, df_prices_xbox_3], axis=1)

df_prices_xbox.reset_index(inplace=True)
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Firmino'], sigma=smoothing_factor), label='Firmino', color='C0')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Mane'], sigma=smoothing_factor), label='Mane', color='C1')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Salah'], sigma=smoothing_factor), label='Salah', color='C2')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Firmino vs Mane vs Salah prices on PS4')

plt.gcf().autofmt_xdate()

plt.show()
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Firmino'], sigma=smoothing_factor), label='Firmino', color='C9')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Mane'], sigma=smoothing_factor), label='Mane', color='C8')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Salah'], sigma=smoothing_factor), label='Salah', color='C7')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Firmino vs Mane vs Salah prices on Xbox')

plt.gcf().autofmt_xdate()

plt.show()
df_prices_ps4.columns = ['date', 'Firmino - PS4', 'Mane - PS4', 'Salah - PS4']

df_prices_xbox.columns = ['date', 'Firmino - Xbox', 'Mane - Xbox', 'Salah - Xbox']

df_prices = df_prices_ps4.merge(df_prices_xbox, on='date')

df_prices['Firmino'] = ((df_prices['Firmino - PS4'] / df_prices['Firmino - Xbox']) - 1) * 100

df_prices['Mane'] = ((df_prices['Mane - PS4'] / df_prices['Mane - Xbox']) - 1) * 100

df_prices['Salah'] = ((df_prices['Salah - PS4'] / df_prices['Salah - Xbox']) - 1) * 100

df_prices = df_prices[['date', 'Firmino', 'Mane', 'Salah']]
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Firmino'], sigma=smoothing_factor), label='Firmino', color='C4')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Mane'], sigma=smoothing_factor), label='Mane', color='C2')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Salah'], sigma=smoothing_factor), label='Salah', color='C6')

ax.set_ylabel('Δ PS4 vs Xbox price (in %)')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Firmino vs Mane vs Salah ratio between PS4 and Xbox prices')

plt.gcf().autofmt_xdate()

plt.show()
resource_ids = {'Cavani': 179813,

                'Neymar': 190871,

                'Mbappé': 231747}



column_names = {'pace': 'Pace', 'dribbling': 'Dribble', 'shooting': 'Shoot',

                'passing': 'Pass', 'defending': 'Defend', 'physicality': 'Physic',

                'pace_acceleration': 'Accel', 'pace_sprint_speed': 'Sprint', 'phys_jumping': 'Jump',

                'phys_stamina': 'Stamina', 'phys_strength': 'Strength', 'phys_aggression': 'Aggress',

                'drib_agility': 'Agility', 'drib_balance': 'Balance', 'drib_reactions': 'React',

                'drib_ball_control': 'Ball ctrl', 'drib_dribbling': 'Dribbl', 'drib_composure': 'Compos',

                'shoot_positioning': 'Posit', 'shoot_finishing': 'Finish', 'shoot_shot_power': 'Shot power',

                'shoot_long_shots': 'Long shot', 'shoot_volleys': 'Volleys', 'shoot_penalties': 'Penalty'}



df16_filtered = df16_players[df16_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df16_filtered.rename(columns=column_names, inplace=True)

df16_filtered.loc['Mbappé'] = np.zeros(94) # adjustment required as Mbappé as missing from FIFA 16

df17_filtered = df17_players[df17_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df17_filtered.rename(columns=column_names, inplace=True)

df17_filtered.rename(index={'Mbappe Lottin': 'Mbappé'},inplace=True) # adjustment required as Mbappé as was called Mbappe Lottin in FIFA 17

df18_filtered = df18_players[df18_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df18_filtered.rename(columns=column_names, inplace=True)

df19_filtered = df19_players[df19_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df19_filtered.rename(columns=column_names, inplace=True)

df20_filtered = df20_players[df20_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df20_filtered.rename(columns=column_names, inplace=True)



attribute_columns_filter = ['Pace', 'Dribble', 'Shoot', 'Pass', 'Defend', 'Physic']

df16_attributes = df16_filtered[attribute_columns_filter]

df17_attributes = df17_filtered[attribute_columns_filter]

df18_attributes = df18_filtered[attribute_columns_filter]

df19_attributes = df19_filtered[attribute_columns_filter]

df20_attributes = df20_filtered[attribute_columns_filter]



physical_columns_filter = ['Accel', 'Sprint', 'Jump', 'Stamina', 'Strength', 'Aggress']

df16_physical = df16_filtered[physical_columns_filter]

df17_physical = df17_filtered[physical_columns_filter]

df18_physical = df18_filtered[physical_columns_filter]

df19_physical = df19_filtered[physical_columns_filter]

df20_physical = df20_filtered[physical_columns_filter]



dribbling_columns_filter = ['Agility', 'Balance', 'React', 'Ball ctrl', 'Dribbl', 'Compos']

df16_dribbling = df16_filtered[dribbling_columns_filter]

df17_dribbling = df17_filtered[dribbling_columns_filter]

df18_dribbling = df18_filtered[dribbling_columns_filter]

df19_dribbling = df19_filtered[dribbling_columns_filter]

df20_dribbling = df20_filtered[dribbling_columns_filter]



shooting_columns_filter = ['Posit', 'Finish', 'Shot power', 'Long shot', 'Volleys', 'Penalty']

df16_shooting = df16_filtered[shooting_columns_filter]

df17_shooting = df17_filtered[shooting_columns_filter]

df18_shooting = df18_filtered[shooting_columns_filter]

df19_shooting = df19_filtered[shooting_columns_filter]

df20_shooting = df20_filtered[shooting_columns_filter]
plt.figure(figsize=(20, 12))



attributes = list(df16_attributes)

att_no = len(attributes)

values1 = df16_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 1, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 16', size=11, y=1.1)



attributes = list(df17_attributes)

att_no = len(attributes)

values1 = df17_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 2, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 17', size=11, y=1.1)



attributes = list(df18_attributes)

att_no = len(attributes)

values1 = df18_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 3, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 18', size=11, y=1.1)



attributes = list(df19_attributes)

att_no = len(attributes)

values1 = df19_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 4, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 19', size=11, y=1.1)



attributes = list(df20_attributes)

att_no = len(attributes)

values1 = df20_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 5, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 20', size=11, y=1.1)



attributes = list(df16_physical)

att_no = len(attributes)

values1 = df16_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 6, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df17_physical)

att_no = len(attributes)

values1 = df17_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 7, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df18_physical)

att_no = len(attributes)

values1 = df18_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 8, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df19_physical)

att_no = len(attributes)

values1 = df19_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 9, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df20_physical)

att_no = len(attributes)

values1 = df20_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 10, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df16_dribbling)

att_no = len(attributes)

values1 = df16_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 11, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df17_dribbling)

att_no = len(attributes)

values1 = df17_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 12, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df18_dribbling)

att_no = len(attributes)

values1 = df18_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 13, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df19_dribbling)

att_no = len(attributes)

values1 = df19_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 14, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df20_dribbling)

att_no = len(attributes)

values1 = df20_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 15, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df16_shooting)

att_no = len(attributes)

values1 = df16_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 16, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df17_shooting)

att_no = len(attributes)

values1 = df17_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 17, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df18_shooting)

att_no = len(attributes)

values1 = df18_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 18, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df19_shooting)

att_no = len(attributes)

values1 = df19_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 19, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df20_shooting)

att_no = len(attributes)

values1 = df20_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 20, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)
df_prices_ps4 = pd.concat([

df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

            (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

            (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

            (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

            (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

            (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'ps4']].set_index('date')])

df_prices_xbox = pd.concat([

    df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

                (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

                (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

                (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

                (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

                (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'xbox']].set_index('date')])

if len(list(resource_ids.keys())) == 2:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4_3 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[2]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2, df_prices_ps4_3], axis=1)

df_prices_ps4.reset_index(inplace=True)

if len(list(resource_ids.keys())) == 2:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox_3 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[2]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2, df_prices_xbox_3], axis=1)

df_prices_xbox.reset_index(inplace=True)
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Cavani'], sigma=smoothing_factor), label='Cavani', color='C0')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Neymar'], sigma=smoothing_factor), label='Neymar', color='C1')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Mbappé'], sigma=smoothing_factor), label='Mbappé', color='C2')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Cavani vs Neymar vs Mbappé prices on PS4')

plt.gcf().autofmt_xdate()

plt.show()
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Cavani'], sigma=smoothing_factor), label='Cavani', color='C9')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Neymar'], sigma=smoothing_factor), label='Neymar', color='C8')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Mbappé'], sigma=smoothing_factor), label='Mbappé', color='C7')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Cavani vs Neymar vs Mbappé prices on Xbox')

plt.gcf().autofmt_xdate()

plt.show()
df_prices_ps4.columns = ['date', 'Cavani - PS4', 'Neymar - PS4', 'Mbappé - PS4']

df_prices_xbox.columns = ['date', 'Cavani - Xbox', 'Neymar - Xbox', 'Mbappé - Xbox']

df_prices = df_prices_ps4.merge(df_prices_xbox, on='date')

df_prices['Cavani'] = ((df_prices['Cavani - PS4'] / df_prices['Cavani - Xbox']) - 1) * 100

df_prices['Neymar'] = ((df_prices['Neymar - PS4'] / df_prices['Neymar - Xbox']) - 1) * 100

df_prices['Mbappé'] = ((df_prices['Mbappé - PS4'] / df_prices['Mbappé - Xbox']) - 1) * 100

df_prices = df_prices[['date', 'Cavani', 'Neymar', 'Mbappé']]
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Cavani'], sigma=smoothing_factor), label='Cavani', color='C4')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Neymar'], sigma=smoothing_factor), label='Neymar', color='C2')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Mbappé'], sigma=smoothing_factor), label='Mbappé', color='C6')

ax.set_ylabel('Δ PS4 vs Xbox price (in %)')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Cavani vs Neymar vs Mbappé ratio between PS4 and Xbox prices')

plt.gcf().autofmt_xdate()

plt.show()
resource_ids = {'De Bruyne': 192985,

                'Pogba': 195864,

                'Kante': 215914}



column_names = {'pace': 'Pace', 'dribbling': 'Dribble', 'shooting': 'Shoot',

                'passing': 'Pass', 'defending': 'Defend', 'physicality': 'Physic',

                'pace_acceleration': 'Accel', 'pace_sprint_speed': 'Sprint', 'phys_jumping': 'Jump',

                'phys_stamina': 'Stamina', 'phys_strength': 'Strength', 'phys_aggression': 'Aggress',

                'drib_agility': 'Agility', 'drib_balance': 'Balance', 'drib_reactions': 'React',

                'drib_ball_control': 'Ball ctrl', 'drib_dribbling': 'Dribbl', 'drib_composure': 'Compos',

                'shoot_positioning': 'Posit', 'shoot_finishing': 'Finish', 'shoot_shot_power': 'Shot power',

                'shoot_long_shots': 'Long shot', 'shoot_volleys': 'Volleys', 'shoot_penalties': 'Penalty'}



df16_players.loc[104, 'resource_id'] = 192985 # De Bruyne has a resource_id of 50524633 in FIFA 16, this had to be replaced



df16_filtered = df16_players[df16_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df16_filtered.rename(columns=column_names, inplace=True)

df17_filtered = df17_players[df17_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df17_filtered.rename(columns=column_names, inplace=True)

df18_filtered = df18_players[df18_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df18_filtered.rename(columns=column_names, inplace=True)

df19_filtered = df19_players[df19_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df19_filtered.rename(columns=column_names, inplace=True)

df20_filtered = df20_players[df20_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df20_filtered.rename(columns=column_names, inplace=True)



attribute_columns_filter = ['Pace', 'Dribble', 'Shoot', 'Pass', 'Defend', 'Physic']

df16_attributes = df16_filtered[attribute_columns_filter]

df17_attributes = df17_filtered[attribute_columns_filter]

df18_attributes = df18_filtered[attribute_columns_filter]

df19_attributes = df19_filtered[attribute_columns_filter]

df20_attributes = df20_filtered[attribute_columns_filter]



physical_columns_filter = ['Accel', 'Sprint', 'Jump', 'Stamina', 'Strength', 'Aggress']

df16_physical = df16_filtered[physical_columns_filter]

df17_physical = df17_filtered[physical_columns_filter]

df18_physical = df18_filtered[physical_columns_filter]

df19_physical = df19_filtered[physical_columns_filter]

df20_physical = df20_filtered[physical_columns_filter]



dribbling_columns_filter = ['Agility', 'Balance', 'React', 'Ball ctrl', 'Dribbl', 'Compos']

df16_dribbling = df16_filtered[dribbling_columns_filter]

df17_dribbling = df17_filtered[dribbling_columns_filter]

df18_dribbling = df18_filtered[dribbling_columns_filter]

df19_dribbling = df19_filtered[dribbling_columns_filter]

df20_dribbling = df20_filtered[dribbling_columns_filter]



shooting_columns_filter = ['Posit', 'Finish', 'Shot power', 'Long shot', 'Volleys', 'Penalty']

df16_shooting = df16_filtered[shooting_columns_filter]

df17_shooting = df17_filtered[shooting_columns_filter]

df18_shooting = df18_filtered[shooting_columns_filter]

df19_shooting = df19_filtered[shooting_columns_filter]

df20_shooting = df20_filtered[shooting_columns_filter]
plt.figure(figsize=(20, 12))



attributes = list(df16_attributes)

att_no = len(attributes)

values1 = df16_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 1, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 16', size=11, y=1.1)



attributes = list(df17_attributes)

att_no = len(attributes)

values1 = df17_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 2, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 17', size=11, y=1.1)



attributes = list(df18_attributes)

att_no = len(attributes)

values1 = df18_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 3, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 18', size=11, y=1.1)



attributes = list(df19_attributes)

att_no = len(attributes)

values1 = df19_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 4, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 19', size=11, y=1.1)



attributes = list(df20_attributes)

att_no = len(attributes)

values1 = df20_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 5, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 20', size=11, y=1.1)



attributes = list(df16_physical)

att_no = len(attributes)

values1 = df16_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 6, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df17_physical)

att_no = len(attributes)

values1 = df17_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 7, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df18_physical)

att_no = len(attributes)

values1 = df18_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 8, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df19_physical)

att_no = len(attributes)

values1 = df19_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 9, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df20_physical)

att_no = len(attributes)

values1 = df20_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 10, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df16_dribbling)

att_no = len(attributes)

values1 = df16_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 11, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df17_dribbling)

att_no = len(attributes)

values1 = df17_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 12, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df18_dribbling)

att_no = len(attributes)

values1 = df18_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 13, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df19_dribbling)

att_no = len(attributes)

values1 = df19_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 14, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df20_dribbling)

att_no = len(attributes)

values1 = df20_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 15, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df16_shooting)

att_no = len(attributes)

values1 = df16_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 16, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df17_shooting)

att_no = len(attributes)

values1 = df17_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 17, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df18_shooting)

att_no = len(attributes)

values1 = df18_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 18, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df19_shooting)

att_no = len(attributes)

values1 = df19_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 19, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df20_shooting)

att_no = len(attributes)

values1 = df20_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 20, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)
df16_prices.loc[df16_prices.resource_id == 50524633, 'resource_id'] = 192985 # De Bruyne has a resource_id of 50524633 in FIFA 16, this had to be replaced



df_prices_ps4 = pd.concat([

df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

            (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

            (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

            (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

            (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

            (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'ps4']].set_index('date')])

df_prices_xbox = pd.concat([

    df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

                (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

                (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

                (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

                (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

                (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'xbox']].set_index('date')])

if len(list(resource_ids.keys())) == 2:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4_3 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[2]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2, df_prices_ps4_3], axis=1)

df_prices_ps4.reset_index(inplace=True)

if len(list(resource_ids.keys())) == 2:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox_3 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[2]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2, df_prices_xbox_3], axis=1)

df_prices_xbox.reset_index(inplace=True)
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['De Bruyne'], sigma=smoothing_factor), label='De Bruyne', color='C0')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Pogba'], sigma=smoothing_factor), label='Pogba', color='C1')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Kante'], sigma=smoothing_factor), label='Kante', color='C2')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - De Bruyne vs Pogba vs Kante prices on PS4')

plt.gcf().autofmt_xdate()

plt.show()
sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['De Bruyne'], sigma=smoothing_factor), label='De Bruyne', color='C9')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Pogba'], sigma=smoothing_factor), label='Pogba', color='C8')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Kante'], sigma=smoothing_factor), label='Kante', color='C7')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - De Bruyne vs Pogba vs Kante prices on Xbox')

plt.gcf().autofmt_xdate()

plt.show()
df_prices_ps4.columns = ['date', 'De Bruyne - PS4', 'Pogba - PS4', 'Kante - PS4']

df_prices_xbox.columns = ['date', 'De Bruyne - Xbox', 'Pogba - Xbox', 'Kante - Xbox']

df_prices = df_prices_ps4.merge(df_prices_xbox, on='date')

df_prices['De Bruyne'] = ((df_prices['De Bruyne - PS4'] / df_prices['De Bruyne - Xbox']) - 1) * 100

df_prices['Pogba'] = ((df_prices['Pogba - PS4'] / df_prices['Pogba - Xbox']) - 1) * 100

df_prices['Kante'] = ((df_prices['Kante - PS4'] / df_prices['Kante - Xbox']) - 1) * 100

df_prices = df_prices[['date', 'De Bruyne', 'Pogba', 'Kante']]
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['De Bruyne'], sigma=smoothing_factor), label='De Bruyne', color='C4')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Pogba'], sigma=smoothing_factor), label='Pogba', color='C2')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Kante'], sigma=smoothing_factor), label='Kante', color='C6')

ax.set_ylabel('Δ PS4 vs Xbox price (in %)')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - De Bruyne vs Pogba vs Kante ratio between PS4 and Xbox prices')

plt.gcf().autofmt_xdate()

plt.show()
resource_ids = {'Chiellini': 138956,

                'Sergio Ramos': 155862,

                'Van Dijk': 203376}



column_names = {'pace': 'Pace', 'dribbling': 'Dribble', 'shooting': 'Shoot',

                'passing': 'Pass', 'defending': 'Defend', 'physicality': 'Physic',

                'pace_acceleration': 'Accel', 'pace_sprint_speed': 'Sprint', 'phys_jumping': 'Jump',

                'phys_stamina': 'Stamina', 'phys_strength': 'Strength', 'phys_aggression': 'Aggress',

                'drib_agility': 'Agility', 'drib_balance': 'Balance', 'drib_reactions': 'React',

                'drib_ball_control': 'Ball ctrl', 'drib_dribbling': 'Dribbl', 'drib_composure': 'Compos',

                'shoot_positioning': 'Posit', 'shoot_finishing': 'Finish', 'shoot_shot_power': 'Shot power',

                'shoot_long_shots': 'Long shot', 'shoot_volleys': 'Volleys', 'shoot_penalties': 'Penalty'}



df16_players.loc[1043, 'resource_id'] = 203376 # Van Dijk has a resource_id of 50535024 in FIFA 16, this had to be replaced



df16_filtered = df16_players[df16_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df16_filtered.rename(columns=column_names, inplace=True)

df17_filtered = df17_players[df17_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df17_filtered.rename(columns=column_names, inplace=True)

df18_filtered = df18_players[df18_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df18_filtered.rename(columns=column_names, inplace=True)

df18_filtered.rename(index={'Ramos': 'Sergio Ramos'},inplace=True) # adjustment required as Sergio Ramos as was called Ramos in FIFA 18-20

df19_filtered = df19_players[df19_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df19_filtered.rename(columns=column_names, inplace=True)

df19_filtered.rename(index={'Ramos': 'Sergio Ramos'},inplace=True) # adjustment required as Sergio Ramos as was called Ramos in FIFA 18-20

df20_filtered = df20_players[df20_players['resource_id'].isin(resource_ids.values())].set_index('player_name').sort_values(by='resource_id', ascending=True)

df20_filtered.rename(columns=column_names, inplace=True)

df20_filtered.rename(index={'Ramos': 'Sergio Ramos'},inplace=True) # adjustment required as Sergio Ramos as was called Ramos in FIFA 18-20



attribute_columns_filter = ['Pace', 'Dribble', 'Shoot', 'Pass', 'Defend', 'Physic']

df16_attributes = df16_filtered[attribute_columns_filter]

df17_attributes = df17_filtered[attribute_columns_filter]

df18_attributes = df18_filtered[attribute_columns_filter]

df19_attributes = df19_filtered[attribute_columns_filter]

df20_attributes = df20_filtered[attribute_columns_filter]



physical_columns_filter = ['Accel', 'Sprint', 'Jump', 'Stamina', 'Strength', 'Aggress']

df16_physical = df16_filtered[physical_columns_filter]

df17_physical = df17_filtered[physical_columns_filter]

df18_physical = df18_filtered[physical_columns_filter]

df19_physical = df19_filtered[physical_columns_filter]

df20_physical = df20_filtered[physical_columns_filter]



dribbling_columns_filter = ['Agility', 'Balance', 'React', 'Ball ctrl', 'Dribbl', 'Compos']

df16_dribbling = df16_filtered[dribbling_columns_filter]

df17_dribbling = df17_filtered[dribbling_columns_filter]

df18_dribbling = df18_filtered[dribbling_columns_filter]

df19_dribbling = df19_filtered[dribbling_columns_filter]

df20_dribbling = df20_filtered[dribbling_columns_filter]



shooting_columns_filter = ['Posit', 'Finish', 'Shot power', 'Long shot', 'Volleys', 'Penalty']

df16_shooting = df16_filtered[shooting_columns_filter]

df17_shooting = df17_filtered[shooting_columns_filter]

df18_shooting = df18_filtered[shooting_columns_filter]

df19_shooting = df19_filtered[shooting_columns_filter]

df20_shooting = df20_filtered[shooting_columns_filter]
plt.figure(figsize=(20, 12))



attributes = list(df16_attributes)

att_no = len(attributes)

values1 = df16_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 1, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 16', size=11, y=1.1)



attributes = list(df17_attributes)

att_no = len(attributes)

values1 = df17_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 2, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 17', size=11, y=1.1)



attributes = list(df18_attributes)

att_no = len(attributes)

values1 = df18_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 3, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 18', size=11, y=1.1)



attributes = list(df19_attributes)

att_no = len(attributes)

values1 = df19_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 4, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 19', size=11, y=1.1)



attributes = list(df20_attributes)

att_no = len(attributes)

values1 = df20_attributes.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_attributes.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_attributes.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 5, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_attributes.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_attributes.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_attributes.index[2], color="C2", size=12)

plt.title('FIFA 20', size=11, y=1.1)



attributes = list(df16_physical)

att_no = len(attributes)

values1 = df16_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 6, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df17_physical)

att_no = len(attributes)

values1 = df17_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 7, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df18_physical)

att_no = len(attributes)

values1 = df18_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 8, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df19_physical)

att_no = len(attributes)

values1 = df19_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 9, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df20_physical)

att_no = len(attributes)

values1 = df20_physical.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_physical.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_physical.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 10, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_physical.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_physical.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_physical.index[2], color="C2", size=12)



attributes = list(df16_dribbling)

att_no = len(attributes)

values1 = df16_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 11, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df17_dribbling)

att_no = len(attributes)

values1 = df17_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 12, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df18_dribbling)

att_no = len(attributes)

values1 = df18_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 13, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df19_dribbling)

att_no = len(attributes)

values1 = df19_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 14, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df20_dribbling)

att_no = len(attributes)

values1 = df20_dribbling.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_dribbling.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_dribbling.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 15, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_dribbling.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_dribbling.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_dribbling.index[2], color="C2", size=12)



attributes = list(df16_shooting)

att_no = len(attributes)

values1 = df16_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df16_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df16_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 16, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df17_shooting)

att_no = len(attributes)

values1 = df17_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df17_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df17_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 17, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df18_shooting)

att_no = len(attributes)

values1 = df18_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df18_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df18_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 18, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df19_shooting)

att_no = len(attributes)

values1 = df19_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df19_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df19_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 19, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)



attributes = list(df20_shooting)

att_no = len(attributes)

values1 = df20_shooting.iloc[0].tolist()

values1 += values1[:1]

angles1 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles1 += angles1[:1]

values2 = df20_shooting.iloc[1].tolist()

values2 += values2[:1]

angles2 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles2 += angles2[:1]

values3 = df20_shooting.iloc[2].tolist()

values3 += values3[:1]

angles3 = [n / float(att_no) * 2 * pi for n in range(att_no)]

angles3 += angles3[:1]

ax = plt.subplot(4, 5, 20, polar=True)

plt.xticks(angles1[:-1], attributes, color='grey', size=10)

plt.yticks([25,50,75], ["25","50","75"], color="grey", size=8)

plt.ylim(0,100)

plt.subplots_adjust(wspace=0.35)

ax.plot(angles1, values1, linewidth=1, linestyle='solid')

ax.fill(angles1, values1, 'C0', alpha=0.1)

ax.plot(angles2, values2, linewidth=1, linestyle='solid')

ax.fill(angles2, values2, 'C1', alpha=0.1)

ax.plot(angles3, values3, linewidth=1, linestyle='solid')

ax.fill(angles3, values3, 'C2', alpha=0.1)

plt.figtext(0.1, 0.98, df16_shooting.index[0], color="C0", size=12)

plt.figtext(0.1, 0.95, df16_shooting.index[1], color="C1", size=12)

plt.figtext(0.1, 0.92, df16_shooting.index[2], color="C2", size=12)
df16_prices.loc[df16_prices.resource_id == 50535024, 'resource_id'] = 203376 # Van Dijk has a resource_id of 50535024 in FIFA 16, this had to be replaced



df_prices_ps4 = pd.concat([

df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

            (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

            (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

            (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

            (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'ps4']].set_index('date'),

df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

            (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'ps4']].set_index('date')])

df_prices_xbox = pd.concat([

    df16_prices[(df16_prices['resource_id'].isin(resource_ids.values())) &

                (df16_prices['date'].between('2015-10-15', '2016-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df17_prices[(df17_prices['resource_id'].isin(resource_ids.values())) &

                (df17_prices['date'].between('2016-10-15', '2017-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df18_prices[(df18_prices['resource_id'].isin(resource_ids.values())) &

                (df18_prices['date'].between('2017-10-15', '2018-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df19_prices[(df19_prices['resource_id'].isin(resource_ids.values())) &

                (df19_prices['date'].between('2018-10-15', '2019-08-15'))][['resource_id' , 'date', 'xbox']].set_index('date'),

    df20_prices[(df20_prices['resource_id'].isin(resource_ids.values())) &

                (df20_prices['date'].between('2019-10-15', end_date))][['resource_id' , 'date', 'xbox']].set_index('date')])

if len(list(resource_ids.keys())) == 2:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_ps4_1 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[0]})

    df_prices_ps4_2 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[1]})

    df_prices_ps4_3 = df_prices_ps4[df_prices_ps4['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'ps4': list(resource_ids.keys())[2]})

    df_prices_ps4 = pd.concat([df_prices_ps4_1, df_prices_ps4_2, df_prices_ps4_3], axis=1)

df_prices_ps4.reset_index(inplace=True)

if len(list(resource_ids.keys())) == 2:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2], axis=1)

elif len(list(resource_ids.keys())) == 3:

    df_prices_xbox_1 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[0]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[0]})

    df_prices_xbox_2 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[1]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[1]})

    df_prices_xbox_3 = df_prices_xbox[df_prices_xbox['resource_id'] == list(resource_ids.values())[2]].drop('resource_id', axis=1).rename(columns = {'xbox': list(resource_ids.keys())[2]})

    df_prices_xbox = pd.concat([df_prices_xbox_1, df_prices_xbox_2, df_prices_xbox_3], axis=1)

df_prices_xbox.reset_index(inplace=True)
sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Chiellini'], sigma=smoothing_factor), label='Chiellini', color='C0')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Sergio Ramos'], sigma=smoothing_factor), label='Sergio Ramos', color='C1')

ax.plot(df_prices_ps4['date'], gaussian_filter1d(df_prices_ps4['Van Dijk'], sigma=smoothing_factor), label='Van Dijk', color='C2')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Chiellini vs Sergio Ramos vs Van Dijk prices on PS4')

plt.gcf().autofmt_xdate()

plt.show()
sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Chiellini'], sigma=smoothing_factor), label='Chiellini', color='C9')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Sergio Ramos'], sigma=smoothing_factor), label='Sergio Ramos', color='C8')

ax.plot(df_prices_xbox['date'], gaussian_filter1d(df_prices_xbox['Van Dijk'], sigma=smoothing_factor), label='Van Dijk', color='C7')

ax.set_ylabel('Price')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Chiellini vs Sergio Ramos vs Van Dijk prices on Xbox')

plt.gcf().autofmt_xdate()

plt.show()
df_prices_ps4.columns = ['date', 'Chiellini - PS4', 'Sergio Ramos - PS4', 'Van Dijk - PS4']

df_prices_xbox.columns = ['date', 'Chiellini - Xbox', 'Sergio Ramos - Xbox', 'Van Dijk - Xbox']

df_prices = df_prices_ps4.merge(df_prices_xbox, on='date')

df_prices['Chiellini'] = ((df_prices['Chiellini - PS4'] / df_prices['Chiellini - Xbox']) - 1) * 100

df_prices['Sergio Ramos'] = ((df_prices['Sergio Ramos - PS4'] / df_prices['Sergio Ramos - Xbox']) - 1) * 100

df_prices['Van Dijk'] = ((df_prices['Van Dijk - PS4'] / df_prices['Van Dijk - Xbox']) - 1) * 100

df_prices = df_prices[['date', 'Chiellini', 'Sergio Ramos', 'Van Dijk']]
sns.set()

sns.set_context("talk")



plt.figure(figsize=(25, 10))

ax = plt.gca()

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Chiellini'], sigma=smoothing_factor), label='Chiellini', color='C4')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Sergio Ramos'], sigma=smoothing_factor), label='Sergio Ramos', color='C2')

ax.plot(df_prices['date'], gaussian_filter1d(df_prices['Van Dijk'], sigma=smoothing_factor), label='Van Dijk', color='C6')

ax.set_ylabel('Δ PS4 vs Xbox price (in %)')

ax.set_xlim([datetime(2015, 9, 30), datetime(2020, 1, 31)])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.legend()

ax.set_title('FIFA 16-20 - Chiellini vs Sergio Ramos vs Van Dijk ratio between PS4 and Xbox prices')

plt.gcf().autofmt_xdate()

plt.show()