# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

%matplotlib inline
df=pd.read_csv(os.path.join(dirname, filename))
df_mega=pd.read_csv(os.path.join(dirname, filename))
df.head()
df.describe()
df.info()
missing_cols=df.columns[df.isna().any()].tolist()
missing_cols
mega = []
for i, row in df.iterrows():
  if(row.is_mega==1):
    mega.append(i)
df.drop(df.index[mega], inplace=True)
df.describe()
single_type=0
dual_type=0
for i in df.index:
    if df["type2"][i]=="None":
        single_type+=1
    else:
        dual_type+=1
print("Number of single type pokemon are {s} and the number of dual type pokemon are {d}".format(s=single_type, d=dual_type))
labels="single type", "dual type"
sizes=[single_type, dual_type]
explode=(0,0.1)
plt.pie(sizes, labels=labels, explode=explode,autopct='%1.1f%%',shadow=True,
        startangle=270)
plt.axis('equal')
plt.title('Dual vs Single type Pokemon')
plt.tight_layout()
plt.show()
ind_count = pd.value_counts(df['type1'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=ind_count.index, y=ind_count, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 12)
ax.set(xlabel='Primary types', ylabel='Count')
ax.set_title('Distribution of Primary Pokemon type')
plt.show()
ind_count=pd.value_counts(df['type2'])
ind_count=ind_count.drop("None")

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=ind_count.index, y=ind_count, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 12)
ax.set(xlabel='Secondary types', ylabel='Count')
ax.set_title('Distribution of Secondary Pokemon type')
plt.show()
type1=pd.value_counts(df['type1'])
type2=ind_count=pd.value_counts(df['type2'])
overall=type1+type2
overall=overall.drop("None")
overall=overall.sort_values(ascending=False)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=overall.index, y=overall, data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 12)
ax.set(xlabel='Overall', ylabel='Count')
ax.set_title('Distribution of Pokemon type(including primary and secondary)')
plt.show()
plt.subplots(figsize=(10, 10))

sns.heatmap(
    df[df['type2']!='None'].groupby(['type1', 'type2']).size().unstack(),
    linewidths=1,
    annot=True,
    cmap="Blues"
)

plt.xticks(rotation=35)
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.set_style("white")
X=df['base_total']
Y=df["capture_rate"]
ax = sns.scatterplot(x=X, y=Y, data=df,
                     hue=df['is_legendary'], alpha=.9, palette="muted")
ax.set(xlabel='Base stats total', ylabel='Catch rate')
ax.set_title('Relationship between Base stats total and Catch rate')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.set_style("white")
X=df['base_total']
Y=df["capture_rate"]
ax = sns.scatterplot(x=X, y=Y, data=df,
                     hue=df['is_mythical'], alpha=.9, palette="muted")
ax.set(xlabel='Base stats total', ylabel='Catch rate')
ax.set_title('Relationship between Base stats total and Catch rate')
plt.show()
df_mega['height_m']=pd.to_numeric(df_mega['height_m'])
df_mega['weight_kg']=pd.to_numeric(df_mega['weight_kg'])
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")
X=df['height_m']
Y=df['weight_kg']
ax = sns.scatterplot(x=X, y=Y, data=df,
                     alpha=.6, palette="muted")
ax.set(xlabel='Height in meters', ylabel='weigth in kg')
ax.set_title('Relationship between Height and Weight')
plt.show()
print("The tallest pokemon are:")
tall=df_mega['name'][df_mega['height_m']>9].tolist()
for i in tall:
    print(i)

print("\nThe heaviest pokemon are:")
heavy=df_mega['name'][df_mega['weight_kg']>900].tolist()
for i in heavy:
    print(i)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")

ax= sns.distplot(df_mega['height_m'], color="y")
ax.set(xlabel='Height in meters')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")

ax= sns.distplot(df_mega['weight_kg'], color="r")
ax.set(xlabel='Weigth in kg')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")
X=df['attack']
Y=df['defense']
ax = sns.scatterplot(x=X, y=Y, data=df,  
                     hue=df['is_legendary'], 
                     alpha=.9, palette="muted")
ax.set(xlabel='Attack stat', ylabel='Defense stat')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")
X=df['attack']
Y=df['defense']
ax = sns.scatterplot(x=X, y=Y, data=df,  
                     hue=df['is_mythical'], 
                     alpha=.9, palette="muted")
ax.set(xlabel='Attack stat', ylabel='Defense stat')
plt.show()
print("The pokemon with highest attack stat are:")
tall=df['name'][df['attack']>175].tolist()
for i in tall:
    print(i)

print("\nThe pokemon with highest defense stat are:")
heavy=df['name'][df['defense']>200].tolist()
for i in heavy:
    print(i)

print("The pokemon with highest attack stat are:")
tall=df_mega['name'][df_mega['attack']>175].tolist()
for i in tall:
    print(i)

print("\nThe pokemon with highest defense stat are:")
heavy=df_mega['name'][df_mega['defense']>200].tolist()
for i in heavy:
    print(i)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")
X=df['sp_attack']
Y=df['sp_defense']
ax = sns.scatterplot(x=X, y=Y, data=df,  
                     hue=df['is_legendary'], 
                     alpha=.9, palette="muted")
ax.set(xlabel='Special Attack stat', ylabel='Special Defence stat')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")
X=df['sp_attack']
Y=df['sp_defense']
ax = sns.scatterplot(x=X, y=Y, data=df,  
                     hue=df['is_mythical'], 
                     alpha=.9, palette="muted")
ax.set(xlabel='Special Attack stat', ylabel='Special Defence stat')
plt.show()
print("The pokemon with highest special attack stat are:")
tall=df['name'][df['sp_attack']>175].tolist()
for i in tall:
    print(i)

print("\nThe pokemon with highest special defense stat are:")
heavy=df['name'][df['sp_defense']>190].tolist()
for i in heavy:
    print(i)

print("The pokemon with highest special attack stat are:")
tall=df_mega['name'][df_mega['sp_attack']>175].tolist()
for i in tall:
    print(i)

print("\nThe pokemon with highest special defense stat are:")
heavy=df_mega['name'][df_mega['sp_defense']>190].tolist()
for i in heavy:
    print(i)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")

ax= sns.distplot(df_mega['speed'], color="g")
ax.set(xlabel='Speed')
plt.show()
mean= df_mega['speed'].describe()[1]
sd= df_mega['speed'].describe()[2]
fast_pokemon=[]
slow_pokemon=[]
very_fast_pokemon=[]
very_slow_pokemon=[]
normal_pokemon=[]
for i in df_mega.index:
    if(df_mega.speed[i]>mean+(2*sd)):
        very_fast_pokemon.append(df_mega['name'][i])
    elif(df_mega.speed[i]<mean-(2*sd)):
        very_slow_pokemon.append(df_mega['name'][i])
    elif(df_mega.speed[i]>mean+sd):
         fast_pokemon.append(df_mega['name'][i])
    elif(df_mega.speed[i]<mean-sd):
         slow_pokemon.append(df_mega['name'][i])
    else:
         normal_pokemon.append(df_mega['name'][i])
speed_levels=['fast','slow','very fast',
              'very slow' ,'normal']
speed_count = [len(fast_pokemon), len(slow_pokemon), len(very_fast_pokemon)
               ,len(very_slow_pokemon),len(normal_pokemon)]
plt.bar(speed_levels,speed_count)
plt.show()
print('Fastest Pokemon: {}'.format(df_mega.name[df_mega['speed'].idxmax()] ))
print('Slowest Pokemon: {}'.format(df_mega.name[df_mega['speed'].idxmin()] ))
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("white")

ax= sns.distplot(df_mega['hp'], color="b")
ax.set(xlabel='HP')
mean= df_mega['hp'].describe()[1]
sd= df_mega['hp'].describe()[2]
bulky=[]
weak=[]
tank=[]
fragile=[]
normal_pokemon=[]
for i in df_mega.index:
    if(df_mega.speed[i]>mean+(2*sd)):
        tank.append(df_mega['name'][i])
    elif(df_mega.speed[i]<mean-(2*sd)):
        fragile.append(df_mega['name'][i])
    elif(df_mega.speed[i]>mean+sd):
         bulky.append(df_mega['name'][i])
    elif(df_mega.speed[i]<mean-sd):
         weak.append(df_mega['name'][i])
    else:
         normal_pokemon.append(df_mega['name'][i])
hp_levels=['bulky','weak','tank',
              'fragile' ,'normal']
hp_count = [len(bulky), len(weak), len(tank),len(fragile),
               len(normal_pokemon)]
plt.bar(hp_levels,hp_count)
plt.show()
print('The tankiest Pokemon: {}'.format(df_mega.name[df_mega['hp'].idxmax()] ))
print('The most fragile Pokemon: {}'.format(df_mega.name[df_mega['hp'].idxmin()] ))
ax = sns.countplot(x="generation", data=df)
leg=0
non_leg=0
mythical=0
for i in df.index:
    if df['is_legendary'][i]==1:
        leg+=1
    elif df['is_mythical'][i]==1:
        mythical+=1
    else:
        non_leg+=1
labels="non-legendary", "legendary", "mythical"
sizes=[non_leg, leg, mythical]
explode=(0,0.1,0.1)
plt.pie(sizes, labels=labels, explode=explode,autopct='%1.1f%%',shadow=True,
        startangle=90)
plt.axis('equal')
plt.title('Legendary vs Mythical vs Non-legendary pokemon')
plt.tight_layout()
plt.show()
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="generation", hue ='is_legendary', data=df )
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="generation", hue ='is_mythical', data=df )
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="generation", hue ='is_mega', data=df_mega)
corr = df[['hp','attack','sp_attack','defense','sp_defense','speed','base_happiness',  
            'base_egg_steps','capture_rate','is_legendary', 'is_mythical']].corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()
corr = df[['capture_rate','base_happiness','base_egg_steps','is_legendary']].corr()
print (corr)
mixed_type =[]
for i, row in df_mega.iterrows():
    if df_mega.type2[i]=="None":
        mixed_type.append(df_mega['type1'][i])
    else:
        mixed_type.append(df_mega['type1'][i]+" "+df_mega['type2'][i])
df_mega['mixed_type']=mixed_type
df_mega['mixed_type'].value_counts().reset_index(name="count").query("count > 10")["index"]
against_ = ['against_bug', 'against_dark', 'against_dragon', 'against_electric', 'against_fairy', 'against_fighting', 
            'against_fire','against_flying', 'against_ghost', 'against_grass', 'against_ground', 'against_ice', 
            'against_normal','against_poison', 'against_psychic', 'against_rock', 'against_steel', 'against_water']
df_mega['against_aggregate'] = df_mega[against_].sum(axis=1)
df_mega['against_mean'] = df_mega[against_].mean(axis=1)
against_unique = df_mega['against_mean'].unique().tolist()
high = min(against_unique)
low  = max(against_unique)

print ('The pokemon with most resistance to other types')
print (df_mega.name[df_mega['against_mean'] == high])
print ('The pokemon with least resistance to other types')
print (df_mega.name[df_mega['against_mean'] == low])