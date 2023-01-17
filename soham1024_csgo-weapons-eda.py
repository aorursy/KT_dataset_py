import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
weapons= pd.read_csv('/kaggle/input/csgo-weapons/CSGO-Weapons-Data.csv')

skins= pd.read_csv('/kaggle/input/csgo-weapons/weapon_skins_data.csv')
weapons.info()
weapons.head()
weapons['Cost'] = weapons['Cost'].str.replace('$', '')

weapons['Cost'] = weapons['Cost'].astype(int)



weapons['Kill Award'] = weapons['Kill Award'].str.replace('$', '')

weapons['Kill Award'] = weapons['Kill Award'].astype(int)



weapons['Armor Penetration'] = weapons['Armor Penetration'].str.replace('%', '')

weapons['Armor Penetration'] = weapons['Armor Penetration'].astype(float)
weapons[['Cost', 'Kill Award', 'Kills To Rebuy', 'Max Speed',

       'RoF', 'Damage', 'DPS', 'Armor Penetration', 'Penetration', 'Max Range',

       'Clip Size', 'Max Ammo', 'Reload (CR)', 'Reload (FR)', 'Recoil',

       'Recoil (V)', 'Recoil (H)', 'Spread', 'Spread Run', 'Spread Stand',

       'Spread Crouch']]=  weapons[['Cost', 'Kill Award', 'Kills To Rebuy', 'Max Speed',

       'RoF', 'Damage', 'DPS', 'Armor Penetration', 'Penetration', 'Max Range',

       'Clip Size', 'Max Ammo', 'Reload (CR)', 'Reload (FR)', 'Recoil',

       'Recoil (V)', 'Recoil (H)', 'Spread', 'Spread Run', 'Spread Stand',

       'Spread Crouch']].astype(float)
weapons.head()
T_weapons = weapons[weapons['Team']=='T']

CT_weapons = weapons[weapons['Team']=='CT']
T_weapons.head()
columns = ['Cost', 'Kill Award', 'Kills To Rebuy', 'Max Speed',

       'RoF', 'Damage', 'DPS', 'Armor Penetration', 'Penetration', 'Max Range',

       'Clip Size', 'Max Ammo', 'Reload (CR)', 'Reload (FR)', 'Recoil',

       'Recoil (V)', 'Recoil (H)', 'Spread', 'Spread Run', 'Spread Stand',

       'Spread Crouch']
import plotly.express as px



for i in range(0, len(columns)):

    fig = px.bar(T_weapons, x='Name', y=columns[i], text=columns[i])

    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig.show()
for i in range(0, len(columns)):

    fig = px.bar(CT_weapons, x='Name', y=columns[i], text=columns[i])

    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig.show()
skins.head()
skins.fillna(-99.00, inplace=True)