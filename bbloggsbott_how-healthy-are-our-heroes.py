import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
hero_info = pd.read_csv('../input/heroes_information.csv')
hero_info.head(10)
hero_info[(hero_info['Weight'] < 0)].head(10)
hero_pow = pd.read_csv('../input/super_hero_powers.csv')
hero_pow.head()
hero_pow = hero_pow*1
hero_pow.head()
hero_info.shape[0] == hero_pow.shape[0]
names = []
weights = []
agility = []
stamina = []
total_abilities = []
gender = []
height = []
alignments = []
publisher = []
for name, agi, sta, gen, ali, pub in zip(hero_pow['hero_names'], hero_pow['Agility'], hero_pow['Stamina'], hero_info['Gender'], hero_info['Alignment'], hero_info['Publisher']):
    w = hero_info[hero_info['name'] == name]['Weight'].values
    h = hero_info[hero_info['name'] == name]['Height'].values
    abilities = sum(hero_pow[hero_pow['hero_names']==name].iloc[:,1:].values[0])
    if w.shape[0] != 0:
        names.extend([name])
        total_abilities.extend([abilities])
        weights.extend([sum(w)/w.shape[0]])
        agility.extend(['Agile' if agi == 1 else 'Not Agile'])
        stamina.extend(['Has Stamina' if sta == 1 else 'No Stamina'])
        gender.extend([gen])
        height.extend([sum(h)/h.shape[0]])
        alignments.extend([ali])
        publisher.extend([pub])
weights = np.array(weights)
height = np.array(height)
filtered = pd.DataFrame()
filtered['Name'] = np.array(names)[(weights > 0) & (height > 0)]
filtered['Weight'] = weights[(weights > 0) & (height > 0)]
filtered['Agility'] = np.array(agility)[(weights > 0) & (height > 0)]
filtered['Stamina'] = np.array(stamina)[(weights > 0) & (height > 0)]
filtered['Total Abilities'] = np.array(total_abilities)[(weights > 0) & (height > 0)]
filtered['Gender'] = np.array(gender)[(weights > 0) & (height > 0)]
filtered['Height'] = np.array(height)[(weights > 0) & (height > 0)]
filtered['Alignment'] = np.array(alignments)[(weights > 0) & (height > 0)]
filtered['Publisher'] = np.array(publisher)[(weights > 0) & (height > 0)]
filtered.head()
plt.figure(figsize = (20,8))
sns.swarmplot(filtered['Agility'], filtered['Weight'], hue = filtered['Stamina'], palette="Set2", dodge=True)
print(filtered['Name'][filtered['Weight']==max(filtered['Weight'])])
print('Unique Genders in Dataset: {}'.format(np.unique(filtered['Gender'])))
filtered[filtered['Gender']=='-']
_females = ('Mockingbird','Goblin Queen',)
def fill_missing(x):
    if(x['Gender']=='-'):
        if(x['Name'] in _females):
            return 'Female'
        else:
            return 'Male'
    else:
        return x['Gender']

filtered['Gender'] = filtered.apply(fill_missing, axis=1)
print('Unique Genders in Dataset: {}'.format(np.unique(filtered['Gender'])))
sns.countplot(filtered['Gender'][filtered['Publisher']=='Marvel Comics'])
plt.title('Gender count - Marvel Comics')
sns.countplot(filtered['Gender'][filtered['Publisher']=='DC Comics'])
plt.title('Gender Count - DC comics')
#plt.figure(figsize = (20,8))
sns.jointplot(x=filtered['Weight'], y=filtered['Height'], kind = 'reg')
filtered.head()
filtered['BMI'] = np.divide(filtered['Weight'], np.square(filtered['Height']/100))
fig = ff.create_distplot([filtered['BMI'][(filtered['Alignment'] == 'good')  & (filtered['Gender'] == 'Male') & (filtered['BMI'] < 80)], filtered['BMI'][(filtered['Alignment'] == 'good')  & (filtered['Gender'] == 'Female') & (filtered['BMI'] < 80)]], ['BMI- Good, Male', 'BMI - Good, Female'])
fig['layout'].update(title='Distribution of BMI - Good', xaxis=dict(title='BMI'))
py.iplot(fig, filename='Basic Distplot')
fig = ff.create_distplot([filtered['BMI'][(filtered['Alignment'] == 'bad')  & (filtered['Gender'] == 'Male')], filtered['BMI'][(filtered['Alignment'] == 'bad')  & (filtered['Gender'] == 'Female')]], ['BMI- Bad, Male', 'BMI - Bad, Female'])
fig['layout'].update(title='Distribution of BMI - Bad', xaxis=dict(title='BMI'))
py.iplot(fig, filename='Basic Distplot')
filtered.sort_values(['BMI'], ascending=False).head(10)