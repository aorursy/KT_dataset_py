import pandas as pd
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

# https://stackoverflow.com/a/42392805
# https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

%matplotlib inline
# Les sources sont disponibles ici: https://developer.riotgames.com/docs/lol#data-dragon_champions
with open('../input/champions-10.14.1.json') as json_file:
    champions = json.load(json_file)
    all_champions_info = pd.DataFrame.from_dict(champions['data'], orient='index')


all_champions_info['main_role'] = all_champions_info['tags'].apply(lambda x: x[0])
all_champions_info['secondary_role'] = all_champions_info['tags'].apply(lambda x: x[1] if len(x) > 1 else None)
champions = all_champions_info.drop(['tags', 'key', 'title', 'blurb', 'image', 'version', 'info', 'stats'], axis=1)
champions.set_index('id', inplace=True)

champions.head(8)
import pandas_profiling as pp

profile = pp.ProfileReport(champions, title="Analyse des champions de League of Legends");
profile
sns.countplot(x='main_role', data = champions);
# Répartition des champions selon leur rôle principal et secondaire
champs_with_two_roles = champions[champions['secondary_role'].isnull() == False]

sns.countplot(x='main_role', hue='secondary_role', data = champs_with_two_roles);
# Récupération des statistiques des champions
stats = pd.DataFrame.from_dict(all_champions_info['stats'].to_list())
stats['id'] = all_champions_info['id'].to_list()
stats.set_index('id', inplace=True)

# Capacité à prendre des dégats, des niveaux 1 à 18
stats['main_role'] = all_champions_info['main_role']

stats['tankiness_lvl1'] = stats['hp'] + stats['armor'] + stats['spellblock']
stats['tankiness_lvl6'] = (stats['hp'] + stats['hpperlevel'] * 6) + (stats['armor'] + stats['armorperlevel'] * 6) + (stats['spellblock'] + stats['spellblockperlevel'] * 6)
stats['tankiness_lvl11'] = (stats['hp'] + stats['hpperlevel'] * 11) + (stats['armor'] + stats['armorperlevel'] * 11) + (stats['spellblock'] + stats['spellblockperlevel'] * 11)
stats['tankiness_lvl16'] = (stats['hp'] + stats['hpperlevel'] * 16) + (stats['armor'] + stats['armorperlevel'] * 16) + (stats['spellblock'] + stats['spellblockperlevel'] * 16)
stats['tankiness_lvl18'] = (stats['hp'] + stats['hpperlevel'] * 18) + (stats['armor'] + stats['armorperlevel'] * 18) + (stats['spellblock'] + stats['spellblockperlevel'] * 18)

tankiness_stats = stats[['tankiness_lvl1', 'tankiness_lvl6', 'tankiness_lvl11', 'tankiness_lvl16', 'tankiness_lvl18', 'main_role']]
tankiness_stats.head(5)
lvl1_ranking = tankiness_stats.sort_values('tankiness_lvl1', ascending=False)[['tankiness_lvl1', 'main_role']]
lvl6_ranking = tankiness_stats.sort_values('tankiness_lvl6', ascending=False)[['tankiness_lvl6', 'main_role']]
lvl11_ranking = tankiness_stats.sort_values('tankiness_lvl11', ascending=False)[['tankiness_lvl11', 'main_role']]
lvl16_ranking = tankiness_stats.sort_values('tankiness_lvl16', ascending=False)[['tankiness_lvl16', 'main_role']]
lvl18_ranking = tankiness_stats.sort_values('tankiness_lvl18', ascending=False)[['tankiness_lvl18', 'main_role']]

tankiness_stats.head()

lvl6_ranking.head(10)
lvl11_ranking.head(10)
lvl16_ranking.head(10)
lvl18_ranking.head(10)