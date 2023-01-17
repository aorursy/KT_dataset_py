import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')



# Allow several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file = '/kaggle/input/fifa-20-complete-player-dataset/players_20.csv'

df = pd.read_csv(file)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

df.head()
df.shape

df.describe().T
print("There are {} players in this dataset".format(df.sofifa_id.nunique()))

print('Age:')

print("  The average age is {:.2f}, \n  half of the the players are younger than {}, the oldest player is {}, the youngest is {}, \n  the age span between 1st and 3rd quartiles is {} years".format(

    df.age.mean(), df.age.median(), df.age.max(), df.age.min(), df.age.quantile(0.75) - df.age.quantile(.25)))

print('Height/Weight:')

print("  The average height is {:.2f} cm ({} feet {:.2f} in); the average weight is {:.2f} kg ({:.2f} lbs)".format(df.height_cm.mean(), int(df.height_cm.mean()*0.0328), df.height_cm.mean()*0.394 % 12, df.weight_kg.mean(), df.weight_kg.mean()*2.2))
f,a = plt.subplots(1,2,figsize=(18,6))

f.subplots_adjust(wspace = .4)

sns.violinplot(df.age, inner = 'box', orient='v', ax=a[0])

a[0].set_title('Violin Plot', fontsize=15, fontweight='bold')

a[0].set_xlabel('')



# swarmplot is expensive

sns.swarmplot(y = df.age.sample(frac = 0.02), ax=a[1])

a[1].set_title('Swarm Plot', fontsize=15, fontweight='bold')

a[1].set_xlabel('');
cols = ['height_cm','weight_kg','overall','potential','value_eur','wage_eur']

f,a = plt.subplots(3,2,figsize=(18,20))

f.subplots_adjust(wspace = .4, hspace = .3)

for i,col in enumerate(cols):

    sns.distplot(df[col], ax=a[i//2][i%2])

    a[i//2][i%2].set_title(col, fontsize=15, fontweight='bold')

    a[i//2][i%2].set_xlabel('')

plt.show();
cols = ['nationality','club','preferred_foot','work_rate','body_type','real_face']



f,a = plt.subplots(3,2,figsize=(18,20))

f.subplots_adjust(wspace = .4, hspace = .3)

for i,col in enumerate(cols):

    sns.barplot(df[col].value_counts()[:20], df[col].value_counts().index[:20], ax=a[i//2][i%2])

    a[i//2][i%2].set_title(col, fontsize=15, fontweight='bold')

    a[i//2][i%2].set_xlabel('')

plt.show();
print("There are over {} English soccer players; it is a little surprising that China has {} players. In contrast, India has {} players and Russia has {} players".format(

df[df.nationality == 'England'].sofifa_id.size, df[df.nationality == 'China PR'].sofifa_id.size, df[df.nationality == 'India'].sofifa_id.size, df[df.nationality == 'Russia'].sofifa_id.size))

print("There seems a cap for how many players a club can have. Need to re-investigate.")

print("There are {} right-footed players and {} left-footed players; the ratio is {:.2f}".format(df[df.preferred_foot == 'Right'].sofifa_id.size, df[df.preferred_foot == 'Left'].sofifa_id.size, df[df.preferred_foot == 'Right'].sofifa_id.size/df[df.preferred_foot == 'Left'].sofifa_id.size))

print("Most players act Medium/Medium work rate")

print("Most players has Normal body type; seems someone has a 'PLAYER_BODY_TYPE_25' body type, sounds like a factory series number.\n  He is {}, a {} years old {} player, severing {} club, wearing number {}".format(

df[df.body_type == 'PLAYER_BODY_TYPE_25'].long_name.iloc[0], int(df[df.body_type == 'PLAYER_BODY_TYPE_25'].age), df[df.body_type == 'PLAYER_BODY_TYPE_25'].nationality.iloc[0], df[df.body_type == 'PLAYER_BODY_TYPE_25'].club.iloc[0], int(df[df.body_type == 'PLAYER_BODY_TYPE_25'].team_jersey_number)))

print("Most players do not have 'real_face' attribute")
print(df.club.nunique())

# df.club.value_counts()



f,a = plt.subplots(1,1,figsize=(18,5))

# f.subplots_adjust(wspace = .4, hspace = .3)

sns.distplot(df.club.value_counts())

a.set_title("Distibution of Club sizes", fontsize=15, fontweight='bold')

a.set_xlabel('')

plt.show();
from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



countries = df['nationality'].value_counts()



world_data = pd.DataFrame({

   'name':list(countries.index.tolist()[:5]),

    'lat':[52.35,51.16, 40.46, 46.23, -38.42,],

   'lon':[-1.17,10.45, -3.75, 2.21, -63.62,],

   'Players':list(countries.iloc[:5]),

})



world_data.name.tolist()

# create map and display it

world_map = folium.Map(location=[10, -20], zoom_start=2.4,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Players'], world_data['name']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.01,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases on 30th Jan 2020</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
# 2.1 distribution of players, age, wage, potential per country, per club, per position

#     2.2 relationship between age and potential, age and overall

    

    

print("There are {} countries in the dataset".format(df.nationality.nunique()))

top30 = df.nationality.value_counts()[:30]

top30_df = df[df['nationality'].isin(top30.index.tolist())]

# top30_df.head()

# sns.boxplot(top30_df.age, top30_df.nationality);



cols = ['age','wage_eur','potential','overall']



f,a = plt.subplots(2,2,figsize=(18,20))

f.subplots_adjust(wspace = .4, hspace = .3)

for i,col in enumerate(cols):

    sns.boxplot(top30_df[col], top30_df.nationality, ax=a[i//2][i%2])

    a[i//2][i%2].set_title(col + " per Country", fontsize=15, fontweight='bold')

    a[i//2][i%2].set_xlabel('')

plt.show();
print("There are {} clubs in the dataset".format(df.club.nunique()))

random30 = df.club.value_counts().sample(30)

# print(df.club.value_counts().value_counts())

random30_df = df[df['club'].isin(random30.index.tolist())]



cols = ['age','wage_eur','potential','overall']



f,a = plt.subplots(2,2,figsize=(18,20))

f.subplots_adjust(wspace = .4, hspace = .3)

for i,col in enumerate(cols):

    sns.boxplot(random30_df[col], random30_df.club, ax=a[i//2][i%2])

    a[i//2][i%2].set_title(col + " per Club", fontsize=15, fontweight='bold')

    a[i//2][i%2].set_xlabel('')

plt.show();
f,a = plt.subplots(2,1,figsize=(18,20))

f.subplots_adjust(wspace = .4, hspace = .3)

sns.boxplot(df.age, df.potential, orient='v', ax=a[0])

a[0].set_title("Potential per Age", fontsize=15, fontweight='bold')

a[0].set_xlabel('')



sns.boxplot(df.age, df.overall, orient='v', ax=a[1])

a[1].set_title("Overall per Age", fontsize=15, fontweight='bold')

a[1].set_xlabel('')

plt.show();

# sns.boxplot(df.potential, df.age, orient='h');



f,a = plt.subplots(1,1,figsize=(18,5))

# f.subplots_adjust(wspace = .4, hspace = .3)

sns.distplot(df.age)

a.set_title("Distibution of Age", fontsize=15, fontweight='bold')

a.set_xlabel('')

plt.show();
f,a = plt.subplots(1,1,figsize=(18,5))

# f.subplots_adjust(wspace = .4, hspace = .3)

sns.distplot(df.age)

a.set_title("Distibution of Age", fontsize=15, fontweight='bold')

a.set_xlabel('')

plt.show();
df.isnull().sum()[df.isnull().sum() != 0]
pivoted = pd.pivot_table(df, values='overall', columns='nationality', index='age')

pivoted.plot(figsize=(16,10))