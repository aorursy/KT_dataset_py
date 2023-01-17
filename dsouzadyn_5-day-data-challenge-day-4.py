import pandas as pd

import seaborn as sns



# Get our data

digimon_list = pd.read_csv('../input/DigiDB_digimonlist.csv')

# Get a summary of the data

digimon_list.describe()
# Print the first few rows

digimon_list.head()
sns.countplot(digimon_list['Type']).set_title('Digimons Types')
sns.regplot(x=digimon_list['Lv 50 HP'], y=digimon_list['Lv50 SP'], fit_reg=False)
digi_type = digimon_list['Type']

digi_attack = digimon_list['Lv50 Atk']

digi_attr = digimon_list['Attribute']
df = pd.DataFrame({'type': digi_type, 'attribute': digi_attr, 'attack': digi_attack})
df_wide = df.pivot_table(index='type', columns='attribute', values='attack')
sns.heatmap(df_wide)