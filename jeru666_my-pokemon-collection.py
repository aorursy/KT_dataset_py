# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_pokemon = pd.read_csv('../input/pokemon.csv')

print (df_pokemon.shape)
df_pokemon.isnull().values.any()
cols_missing_val = df_pokemon.columns[df_pokemon.isnull().any()].tolist()
print(cols_missing_val)
for col in cols_missing_val:
    print("%s : %d" % (col, df_pokemon[col].isnull().sum()))
msno.bar(df_pokemon[cols_missing_val],figsize=(20,8),color="#32885e",fontsize=18,labels=True,)
msno.matrix(df_pokemon[cols_missing_val],width_ratios=(10,1),\
            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)
msno.heatmap(df_pokemon[cols_missing_val],figsize=(10,10))
for col in cols_missing_val:
    print("%s : %d" % (col,df_pokemon[col].nunique()))
df_pokemon['percentage_male'].fillna(np.int(-1), inplace=True)
df_pokemon['type2'].unique()
df_pokemon['type2'].fillna('HHH', inplace=True)
df_pokemon['height_m'].fillna(np.int(0), inplace=True)
df_pokemon['weight_kg'].fillna(np.int(0), inplace=True)
df_pokemon.isnull().values.any()
print(df_pokemon.dtypes.unique())
print(df_pokemon.dtypes.nunique())
pp = pd.value_counts(df_pokemon.dtypes)
pp.plot.bar()
plt.show()
mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(df_pokemon)
mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(df_pokemon)

mem = df_pokemon.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
object_cols = list(df_pokemon.select_dtypes(include=['object']).columns)
df_pokemon = df_pokemon.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('HHH', np.nan)
df_pokemon['type1'].unique()
df_pokemon['type2'].unique()
#--- Snippet to split pokemon based on whether they are of single type or dual type ---
single_type_pokemon = []
dual_type_pokemon = []

count = 0
for i in df_pokemon.index:
    if(pd.isnull(df_pokemon.type2[i]) == True):
    #if(df_pokemon.type2[i] == np.nan):
        count += 1
        single_type_pokemon.append(df_pokemon.name[i])
    else:
        dual_type_pokemon.append(df_pokemon.name[i])

print(len(dual_type_pokemon))
print(len(single_type_pokemon))
data = [417, 384]
colors = ['yellowgreen', 'lightskyblue']

# Create a pie chart
plt.pie(data, 
        labels= ['Dual type', 'Single type'], 
        shadow=True, 
        colors=colors, 
        explode=(0, 0.15), 
        startangle=90, 
        autopct='%1.1f%%')

# View the plot drop above
plt.axis('equal')
plt.title('Dual vs Single type Pokemon')
# View the plot
plt.tight_layout()
plt.show()
yy = pd.value_counts(df_pokemon['type1'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=df_pokemon)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Primary types', ylabel='Count')
ax.set_title('Distribution of Primary Pokemon type')
yy = pd.value_counts(df_pokemon['type2'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=df_pokemon)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Secondary types', ylabel='Count')
ax.set_title('Distribution of Secondary Pokemon type')
df_pokemon['classfication'].nunique()
ss = pd.value_counts(df_pokemon['classfication'])
for i in range(0, 10):
    
    print ("{} : {} ".format(ss.index[i],  ss[i]))
ax = sns.countplot(y=df_pokemon['percentage_male'], data=df_pokemon)  
print ('Purely masculine pokemon : ', len(df_pokemon[df_pokemon.percentage_male == 100.0]))
print ('Purely feminine pokemon : ', len(df_pokemon[df_pokemon.percentage_male == 0.0]))
print ('Genderless pokemon : ', len(df_pokemon[df_pokemon.percentage_male == -1]))

print ('More masculine pokemon : ', len(df_pokemon[df_pokemon.percentage_male > 50.0]))
print ('More feminine pokemon : ', len(df_pokemon[(df_pokemon.percentage_male < 50.0) & (df_pokemon.percentage_male > -1.0)]))

print ('Mixture of feminine & masculine pokemon : ', len(df_pokemon[(df_pokemon.percentage_male < 100.0) & (df_pokemon.percentage_male > 0.0)]))
df_pokemon['capture_rate'].unique()
yy = pd.value_counts(df_pokemon['capture_rate'])

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_style("whitegrid")

ax = sns.barplot(x=yy.index, y=yy, data=df_pokemon)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ax.set(xlabel='Capture_rate', ylabel='Number of Pokemon')
ax.set_title('Distribution of capture_rate against number of Pokemon')
df_pokemon['abilities'].nunique()
df_pokemon['abilities'].head(20).unique()
import re
list_ability = df_pokemon['abilities'].tolist()
len(list_ability)
new_list = []
for i in range(0, len(list_ability)):
    m = re.findall(r"'(.*?)'", list_ability[i], re.DOTALL)
    for j in range(0, len(m)):
        new_list.append(m[j])

list1 = set(new_list)
print(list1)
print(len(list1))



from collections import Counter
count  = pd.Series(df_pokemon['abilities'].str.replace('[\[\]\'\s]','').str.split(',').map(Counter).sum())

print(count.index)
ax = sns.countplot(count)
df_pokemon['pokedex_number'].describe()
ax_height = sns.distplot(df_pokemon['height_m'], color="y")
ax_weight = sns.distplot(df_pokemon['weight_kg'], color="r")
#--- Average weight ---
ax = sns.pointplot(df_pokemon['weight_kg'])
#---Average height ---
ax = sns.pointplot(df_pokemon['height_m'], color = 'g')
df_pokemon['base_egg_steps'].nunique()
ax = sns.countplot(df_pokemon['base_egg_steps'])
df_pokemon['experience_growth'].nunique()
ax = sns.countplot(df_pokemon['experience_growth'])
df_pokemon['base_happiness'].nunique()
ax = sns.countplot(df_pokemon['base_happiness'])
df_pokemon['hp'].nunique()
ax = sns.distplot(df_pokemon['hp'], rug=True, hist=False)
print(df_pokemon['attack'].nunique())
print(df_pokemon['defense'].nunique())
ax_attack = sns.distplot(df_pokemon['attack'], color="r", hist=False)
ax_defense = sns.distplot(df_pokemon['defense'], color="b", hist=False)
print(df_pokemon['sp_attack'].nunique())
print(df_pokemon['sp_defense'].nunique())
ax_attack = sns.distplot(df_pokemon['sp_attack'], color="g", hist=False)
ax_defense = sns.distplot(df_pokemon['sp_defense'], color="y", hist=False)
cols = df_pokemon.columns
against_ = []
for col in cols:
    if ('against_' in str(col)):
        against_.append(col)
        
print(len(against_)) 
print(against_)
unique_elem = []
for col in against_:
    unique_elem.append(df_pokemon[col].unique().tolist())
    
result = set(x for l in unique_elem for x in l)

result = list(result)
print(result)

for col in against_:
    if (np.mean(df_pokemon[col]) > 1.2):
        print(col)

for col in against_:
    if (np.sum(df_pokemon[col]) > 1000):
        print(col)            
import random

for col in range(0, len(against_)):
    print (against_[col])
    print (df_pokemon[against_[col]].unique())
    pp = pd.value_counts(df_pokemon[against_[col]])
    
    color = ['g', 'b', 'r', 'y', 'pink', 'orange', 'brown']
            
    pp.plot.bar(color = random.choice(color))
    plt.show()
print(df_pokemon['speed'].nunique())
df_pokemon['speed'].describe()
ax_height = sns.distplot(df_pokemon['speed'], color="orange")
print('Fastest Pokemon: {}'.format(df_pokemon.name[df_pokemon['speed'].idxmax()] ))
print('Slowest Pokemon: {}'.format(df_pokemon.name[df_pokemon['speed'].idxmin()] ))
speed_statistics = df_pokemon['speed'].describe()

mean = speed_statistics[1]
standard_dev = speed_statistics[2]

#--- Create lists for the four categories mentioned ---
fast_pokemon = []
slow_pokemon = []
v_fast_pokemon = []
v_slow_pokemon = []
normal = []

for i in range(0, len(df_pokemon)):
    if(df_pokemon.speed[i] > mean + (2 * standard_dev)):
        v_fast_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] < mean - (2 * standard_dev)):
        v_slow_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] > mean + standard_dev):
        fast_pokemon.append(df_pokemon.name[i])
    elif(df_pokemon.speed[i] < mean - standard_dev):
        slow_pokemon.append(df_pokemon.name[i])
    else:
        normal.append(df_pokemon.name[i])
    
speed_levels = ['fast_pokemon','slow_pokemon','v_fast_pokemon','v_slow_pokemon','normal']
speed_count = [len(fast_pokemon), len(slow_pokemon), len(v_fast_pokemon),len(v_slow_pokemon),len(normal)]

xlocations = np.array(range(len(speed_count)))
width = 0
plt.bar(xlocations, speed_count, color = 'r')
plt.xticks(xlocations+ width, speed_levels)
#xlim(0, xlocations[-1]+width*2)
plt.title("Count of Pokemon for different Speed levels")
print(df_pokemon['generation'].nunique())
ax = sns.countplot(x="generation", data=df_pokemon)
pp = pd.value_counts(df_pokemon.generation)
pp.plot.pie(startangle=90, autopct='%1.1f%%', shadow=False, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05))
plt.axis('equal')
plt.show()
ax = sns.countplot(y=df_pokemon['is_legendary'], data=df_pokemon, facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
g = sns.jointplot("base_egg_steps", "experience_growth", data=df_pokemon, size=5, ratio=3, color="g")

g = sns.jointplot("base_egg_steps", "experience_growth", data=df_pokemon, kind="kde", space=0, color="g")
g = sns.jointplot("attack", "hp", data=df_pokemon, kind="kde")
g = sns.factorplot(x="attack", y="hp", data=df_pokemon,
                   size=16, kind="bar", palette="muted")
lolo = []
for i in range(0, len(df_pokemon)):
    if(df_pokemon.is_legendary[i] > 0):
        lolo.append(df_pokemon.type1[i])
        
print(set(lolo))  

lolo = []
lulu = []
lulu_attack = []
for i in range(0, len(df_pokemon)):
    if(df_pokemon.is_legendary[i] > 0):
        lolo.append(df_pokemon.experience_growth[i])
        lulu.append(df_pokemon.base_egg_steps[i])
        lulu_attack.append(df_pokemon.attack[i])
        
print(set(lolo))
print(set(lulu))   
print(set(lulu_attack))   

df_pokemon.base_egg_steps.corr(df_pokemon.is_legendary)

numeric_clmns = df_pokemon.dtypes[df_pokemon.dtypes != "object"].index 

f, ax = plt.subplots(figsize=(13, 12))
corr = df_pokemon[numeric_clmns].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sd = df_pokemon[['type1', 'type2', 'is_legendary']]
md = pd.get_dummies(sd)

corr = md.corr()
f, ax = plt.subplots(figsize=(13, 12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df_pokemon.capture_rate.unique()
df_pokemon['capture_rate'].replace('30 (Meteorite)255 (Core)', '1000', inplace=True)
pd.to_numeric(df_pokemon['capture_rate'])
df_pokemon['capture_rate'] = df_pokemon['capture_rate'].astype(int)
df_pokemon.capture_rate.unique()
corr = df_pokemon[['percentage_male', 'capture_rate','defense','sp_defense','base_happiness', 'speed', 'hp','attack','sp_attack','base_egg_steps','experience_growth','is_legendary']].corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
corr = df_pokemon[['percentage_male', 'capture_rate','base_happiness','base_egg_steps','is_legendary']].corr()
print (corr)
ax = sns.countplot(y=df_pokemon['is_legendary'], data=df_pokemon, facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
f, ax = plt.subplots(figsize=(8, 8))
ax = sns.countplot(x="generation", hue = 'is_legendary', data=df_pokemon, )
#--- Replacing missing values with a space character ----
df_pokemon.type2 = df_pokemon.type2.fillna('')
print(df_pokemon.type2.isnull().sum())
df_pokemon['type1_&_type2'] = df_pokemon['type1'] + str(' ') + df_pokemon['type2']
df_pokemon['type1_&_type2'].nunique()
f, ax = plt.subplots(figsize=(15, 9))
ax = sns.countplot(x="type1_&_type2", data=df_pokemon)
df_pokemon['type1_&_type2'].value_counts().reset_index(name="count").query("count > 10")["index"]
df_pokemon['new_type1_&_type2'], _ = pd.factorize(df_pokemon['type1_&_type2'])
corr = df_pokemon[['new_type1_&_type2', 'is_legendary']].corr()
print(corr)
print(len(against_)) 
print(against_)
print(result)
df_pokemon['against_aggregate'] = df_pokemon[against_].sum(axis=1)
df_pokemon['against_mean'] = df_pokemon[against_].mean(axis=1)
df_pokemon.head()
df_pokemon.name[df_pokemon['against_aggregate'].idxmax()]
df_pokemon.name[df_pokemon['against_aggregate'].idxmin()]
against_unique = df_pokemon['against_aggregate'].unique().tolist()
weak = min(against_unique)
strong = max(against_unique)

print ('Strong Pokemon')
print (df_pokemon.name[df_pokemon['against_aggregate'] == strong])
print ('Weak Pokemon')
print (df_pokemon.name[df_pokemon['against_aggregate'] == weak])
corr = df_pokemon[['against_mean', 'is_legendary']].corr()
print(corr)
df_pokemon.groupby('is_legendary', as_index=False)['against_aggregate'].mean()
