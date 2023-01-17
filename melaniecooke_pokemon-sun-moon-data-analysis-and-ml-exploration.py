# Import analysis and visualization packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import data
pokemon = pd.read_csv('../input/pokemon.csv')
movesets = pd.read_csv('../input/movesets.csv')
pokemon.info()
pokemon.iloc[802:808]
pokemon[pokemon['species']=='Pikachu'][['ndex','species','forme','hp','attack','defense','spattack','spdefense','speed','total']]
# Data cleaning
pokemon.drop_duplicates(subset=['species','ndex','hp','attack','defense','spattack','spdefense','speed','total'], keep='first', inplace=True)
# Testing cleanup
print('Testing duplicate forme removal...')
# There should be 1 Pikachu with forme 'Pikachu'
print(pokemon[pokemon['species']=='Pikachu']['forme'] == 'Pikachu')
print(pokemon[pokemon['species']=='Pikachu'].shape[0] == 1)
# There should be 2 Raichu, regular and Alolan forme
print(pokemon[pokemon['species']=='Raichu'].shape[0] == 2)
# There should be 4 Deoxys
print(pokemon[pokemon['species']=='Deoxys'].shape[0] == 4)
# There should be 2 Rotom
print(pokemon[pokemon['species']=='Rotom'].shape[0] == 2)

pokemon[pokemon['species']=='Pikachu'][['ndex','species','forme','hp','attack','defense','spattack','spdefense','speed','total']]
#pokemon.iloc[880:][['id','species','hp','attack','defense','spattack','spdefense','speed','total','forme']]
n_pokemon = pokemon.shape[0]
is_fully_evolved = np.zeros(n_pokemon)
is_mega = np.zeros(n_pokemon)
is_forme = np.zeros(n_pokemon)

for i, species in enumerate(pokemon['species']):
    # Check if pokemon name is found in the pre-evolution list.
    # If it is not, then it must be fully evolved
    if pokemon[pokemon['pre-evolution'].isin([species])].shape[0] == 0:
        is_fully_evolved[i] = 1
        
    if len(pokemon['forme'].iloc[i].split()) > 1:
        if pokemon['forme'].iloc[i].split()[1] == '(Mega':
            is_mega[i] = 1
        
    if pokemon['species'].iloc[i] != pokemon['forme'].iloc[i]:
        is_forme[i] = 1
pokemon['is_fully_evolved'] = is_fully_evolved
pokemon['is_first_stage'] = pokemon['pre-evolution'].isnull()
pokemon['is_mega'] = is_mega
pokemon['is_forme'] = is_forme
pokemon['weight'] = pokemon['weight'].apply(lambda x: float(x.split()[0]))
def height_to_numeric(height):
    height = height.split("'")
    feet = float(height[0])
    inches = float(height[1][:2])
    return feet + (inches/12)
    
pokemon['height'] = pokemon['height'].apply(height_to_numeric)
generation_limits = [151, 251, 386, 493, 649, 721, 807]
def generation(ndex):
    if 1 <= ndex <= 151:
        return 1
    elif 152 <= ndex <= 251:
        return 2
    elif 252 <= ndex <= 386:
        return 3
    elif 387 <= ndex <= 493:
        return 4
    elif 494 <= ndex <= 649:
        return 5
    elif 650 <= ndex <= 721:
        return 6
    elif 722 <= ndex <= 807:
        return 7
pokemon['generation'] = pokemon['ndex'].apply(generation)
for i in range(n_pokemon):
    if len(pokemon['forme'].iloc[i].split()) > 1:
        if pokemon['forme'].iloc[i].split()[1] == '(Alola':
            pokemon['generation'].iloc[i] = 7
        elif pokemon['forme'].iloc[i].split()[1] == '(Mega':
            pokemon['generation'].iloc[i] = 6
pokemon[['ndex','species','forme','is_fully_evolved','is_first_stage','is_mega','is_forme','generation']].iloc[800:820]
movesets.shape
moveset_size = np.zeros(n_pokemon)
for i in range(n_pokemon):
    current_forme = pokemon.iloc[i]['forme']
    # The set of 'formes' in the movesets dataframe sometimes only has the species name (ie 'Burmy')
    # rather than the full forme name (ie 'Burmy (Plant Cloak)'). So we need to check if this is the case
    # and split the forme name and take just its species name. 
    if movesets[movesets['forme'].isin([current_forme])]['forme'].shape[0] == 0:
        current_forme = current_forme.split()[0]
    if movesets[movesets['forme'].isin([current_forme])]['forme'].shape[0] != 0:
        current_set = movesets[movesets['forme']==current_forme]
        moveset_size[i] = current_set.dropna(axis=1).shape[1] - 3

pokemon['moveset_size'] = moveset_size
pokemon[(pokemon['moveset_size']>=120)][['forme','total','moveset_size']]
mean_fe_stats_bygen = pokemon[pokemon['is_fully_evolved']==1].groupby(by='generation').mean()
sns.set_style('darkgrid')
sns.lmplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0], aspect=1.3)
plt.title('Fully evolved Pokemon (excluding Megas)')
plt.ylabel('Base Stat Total')
yt = plt.yticks(range(100,850,50))
fig = plt.figure(figsize=(10,6))
sns.violinplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0],
              palette='Pastel1')
plt.title('Fully evolved Pokemon (excluding megas)')
yl = plt.ylabel('Base Stat Total')
fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='total', data=pokemon[(pokemon['is_fully_evolved']==1) & pokemon['is_mega']==0],
              palette='Pastel1', whis=0.5)
plt.title('Fully evolved Pokemon (excluding megas)')
yl = plt.ylabel('Base Stat Total')
fig = plt.figure(figsize=(10,6))
bins = np.arange(160,800,20)
pokemon[(pokemon['is_first_stage']==1) & (pokemon['is_fully_evolved']==0)]['total'].plot.hist(bins=bins, color='grey', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='First stage')
pokemon[(pokemon['is_fully_evolved']==0) & (pokemon['is_first_stage']==0)]['total'].plot.hist(bins=bins, color='orange', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Middle stage')
pokemon[(pokemon['is_fully_evolved']==1) & (pokemon['is_first_stage']==0) & (pokemon['is_mega']==0)]['total'].plot.hist(bins=bins, color='red', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Fully evolved')
pokemon[(pokemon['is_mega']==1)]['total'].plot.hist(bins=bins, color='blue', edgecolor='black', linewidth=1.2, alpha=0.5, normed=True, label='Mega')
plt.legend()
plt.xlabel('Base Stat Total')
strong_pokemon = pokemon[pokemon['total'] > 440]
fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='total', data=strong_pokemon,
           palette='Pastel1', whis=1)
plt.title('Pokemon with BST exceeding 440')
plt.ylabel('Base Stat Total')
def stat_distribution(hp,attack,defense,spattack,spdefense,speed):
    stat_max = max([hp,attack,defense,spattack,spdefense,speed])
    stat_min = min([hp,attack,defense,spattack,spdefense,speed])
    stat_range = stat_max - stat_min
    return stat_max/stat_min

stat_distr = np.zeros(n_pokemon)
for i in range(n_pokemon):
    stat_distr[i] = stat_distribution(pokemon.hp.iloc[i], pokemon.attack.iloc[i], pokemon.defense.iloc[i],
                                     pokemon.spattack.iloc[i], pokemon.spdefense.iloc[i], pokemon.speed.iloc[i])
pokemon['stat_distribution'] = stat_distr
fig = plt.figure(figsize=(10,6))
sns.boxplot(x='generation', y='stat_distribution', data=pokemon[(pokemon['stat_distribution']<20) & (pokemon['total']>450)],
           palette='Pastel1')
plt.title('Pokemon with BST > 450')
yl = plt.ylabel('Maximum Stat / Minimum Stat')
xl = plt.xlabel('Generation')
yt = plt.yticks(range(1,11,1))
pokemon[(pokemon['stat_distribution']>=7) & (pokemon['is_fully_evolved']==1)][['species','hp','attack','defense','spattack','spdefense','speed']]
fig = plt.figure(figsize=(10,6))
sns.lmplot(x='ndex', y='stat_distribution', data=pokemon[(pokemon['stat_distribution']<20) & (pokemon['total']>450)])
fig = plt.figure(figsize=(9,6))
typecounts = pokemon.groupby(['type1']).count()['total'] + pokemon.groupby(['type2']).count()['total']
cmap = plt.cm.get_cmap('tab20b')
typecounts.sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('count')
typecounts1 = pokemon.groupby(['type1']).count()['total']
typecounts2 = pokemon.groupby(['type2']).count()['total']
sortind = np.argsort(typecounts1)[::-1]
fig, ax = plt.subplots(figsize=(10,6))
index = np.arange(len(typecounts))
bar_width = 0.35
rects1 = ax.bar(index, typecounts1[sortind], bar_width, label='Type 1')
rects2 = ax.bar(index + bar_width, typecounts2[sortind], bar_width, label='Type 2')
xt = plt.xticks(range(len(typecounts)), typecounts.index, rotation=90)
lg = plt.legend()
yl = plt.ylabel('count')
fig = plt.figure(figsize=(9,6))
pokemon[pokemon['is_fully_evolved']==1].groupby('type1').mean()['total'].sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('Average BST')
fig = plt.figure(figsize=(9,6))
pokemon[pokemon['is_fully_evolved']==1].groupby('type2').mean()['total'].sort_values(axis=0, ascending=False).plot(kind='bar', color=cmap(np.linspace(0,1,len(typecounts+1))))
yl = plt.ylabel('Average BST')
fig = plt.figure(figsize=(10,6))
sns.lmplot(x='total',y='moveset_size',data=pokemon[(pokemon['is_fully_evolved']==1)],hue='generation', 
           fit_reg=False, aspect=1.5, size=8, palette='plasma')
yl = plt.ylabel("Moveset Size", fontsize=16)
xl = plt.xlabel('Base Stat Total', fontsize=16)
t = plt.title('Fully evolved Pokemon', fontsize=16)
fig = plt.figure()
sns.lmplot(x='height',y='weight',data=pokemon)
yl = plt.ylabel("Weight (lbs)", fontsize=16)
xl = plt.xlabel('Height (feet)', fontsize=16)
t = plt.title('Pokemon sizes with linear fit', fontsize=16)
pokemon[(pokemon['height']>25)][['forme','height','weight']]
fig = plt.figure(figsize=(10,4))
bins = list(range(0,200,2))#+list(range(50,200,10))
(pokemon['weight']/pokemon['height']).plot.hist(bins=bins, linewidth=1.2, edgecolor='black')
xl = plt.xlabel('Weight / Height (lbs per feet)')
t = plt.title('Pseudo density of Pokemon (excl Cosmoem)')
pokemon[(pokemon['weight']/pokemon['height']) <1.5][['forme','height','weight']]
pokemon[(pokemon['weight']>661) & (pokemon['weight']<881)].sort_values(by='weight', ascending=False)[['forme','weight','height']]
pokemon.columns
pokemonML = pokemon[['hp','attack','defense','spattack','spdefense','speed','total','weight','height','generation','moveset_size']]
X = pokemonML
y = pokemon['is_fully_evolved']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_X)
X_pca = pca.transform(scaled_X)
X_pca.shape
plt.figure(figsize=(10,6))
nfeHandle = plt.scatter(X_pca[(y.values==0),0], X_pca[(y.values==0),1], color='green')
feHandle = plt.scatter(X_pca[(y.values==1),0], X_pca[(y.values==1),1], color='purple')
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.legend(('Not Fully Evolved','Fully Evolved'))
print(pca.explained_variance_ratio_)
first_bool = (pokemon['is_first_stage']==1) & (pokemon['is_fully_evolved']==0)
middle_bool = (pokemon['is_fully_evolved']==0) & (pokemon['is_first_stage']==0)
final_bool = (pokemon['is_fully_evolved']==1) & (pokemon['is_first_stage']==0) & (pokemon['is_mega']==0)
mega_bool = (pokemon['is_mega']==1)

plt.figure(figsize=(12,8))
nfeHandle = plt.scatter(X_pca[(first_bool),0], X_pca[(first_bool),1], color='blue')
middleHandle = plt.scatter(X_pca[(middle_bool),0], X_pca[(middle_bool),1], color='orange')
feHandle = plt.scatter(X_pca[(final_bool),0], X_pca[(final_bool),1], color='purple')
megaHandle = plt.scatter(X_pca[(mega_bool),0], X_pca[(mega_bool),1], color='green')
plt.xlabel('First PC')
plt.ylabel('Second PC')
plt.legend(('First Stage','Middle Stage','Fully Evolved','Mega'))
# Visualize the contributions of each feature to the first 2 PCs
pca_components = pd.DataFrame(pca.components_, columns=X.columns)
plt.figure(figsize=(12,6))
sns.heatmap(pca_components,cmap='PiYG')
plt.yticks((0.5,1.5), ('PC 1', 'PC 2'))
#plt.figure(figsize=(8,6))
#sns.lmplot(x='weight', y='total', data=pokemon)
sns.pairplot(data=X.drop('generation', axis=1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.drop('generation', axis=1), y, test_size=0.30, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=60)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))






