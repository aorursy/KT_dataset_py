import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
# !ln -s ~/data/ data #creating symlink

data = pd.read_csv('../input/pokemon.csv')
s = pd.Series([1, 3, 5, np.nan, 6, 8])
s
d = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
d
pd.DataFrame.from_dict(d)
data.head(3)
data.tail(3)
data.info(True)
data.memory_usage(deep=True)
data.dtypes
#correlation plot
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5)
data.columns  #listing the columns
data.shape  #shape of the dataframe
data.isnull().sum()
data.nunique()
data.sort_values(by='Attack', ascending=False)

# %%timeit -r 3 #-r option to specify number of loops
# data1 = pd.read_csv('data/talking-data/test.csv')
# %%time
# data1 = pd.read_csv('data/talking-data/test.csv')
# %time
# data1.head()
# data1 = pd.read_csv(
#     'data/talking-data/test.csv')  #this won't be evaluated for runtime
# data1.info(verbose=1)
# data1.memory_usage()
# data1.head()
# data1.describe()
# %%time
# dtypes = {
#     'click_id': 'uint32',
#     'ip': 'uint32',
#     'app': 'uint16',
#     'device': 'uint16',
#     'os': 'uint16',
#     'channel': 'uint16'
# }
# data1 = pd.read_csv('data/talking-data/test.csv', dtype=dtypes)
# data1.info(verbose=1)
# %%time
# df = dd.read_csv('data/talking-data/test.csv', dtype=dtypes)
# df = df.compute()  #.compute() converts dask dataframe back to pandas
# df.info(verbose=1)
# %%time
# data1 = pd.read_csv('data/talking-data/test.csv', dtype=dtypes, nrows=100)
# data1.shape
# data1.tail()
# data1 = pd.read_csv(
#     'data/talking-data/test.csv', dtype=dtypes, nrows=100, skiprows=99)
# data1.head()
# data1 = pd.read_csv(
#     'data/talking-data/test.csv',
#     dtype=dtypes,
#     nrows=100,
#     skiprows=range(1, 100))
# data1.head()
# data1 = pd.read_csv(
#     'data/talking-data/test.csv', dtype=dtypes, nrows=100, skiprows=[1, 3, 5])
# data1.head()
data.head()
data['Name'][0:4]  #pandas series
data.Name[0:4]  #we can also use .
data[['Name']][0:4]  #pandas dataframe
data['Type 1'].value_counts()  #can be called on a series and not a dataframe
# if there are nan values that also be counted
data['Type 1'].value_counts(dropna=False)
data.HP.mean()
data.HP.max()
data.HP.count()
data.boxplot(column='Attack', by='Legendary')
data.T
data.describe(include=['number'])
data.select_dtypes(include=['category'])
data.isnull().sum()
data.shape
data.dtypes
data['Name'] = data['Name'].astype('category')
data.dtypes
data.dropna(axis=0, how='any').shape
data.dropna(axis=0, how='all').shape
data['Type 1'].fillna('unknown').isnull().sum()
data.Name.replace('Bulbasaur', 'Bulba')[:4]
data.dropna(axis=0, how='any', inplace=True)
data.isnull().sum()
data.duplicated('Type 1')[:4]  #check for duplicates within a column
data.Legendary.drop_duplicates(
    keep='last')  #remove duplicates and keep the last entry within duplicates
data['Type 1'].unique()
data.loc[1:4, ["HP"]]
data.iloc[1, 1]
data.loc[1:3, ["HP", "Attack"]]
data[["HP", "Attack"]][:3]
data.loc[(data['Defense'] > 200) & (data['Attack'] > 100)]  #and
data.loc[(data['Defense'] > 200) | (data['Attack'] > 100)]  #or
my_pokemon = ['Volcanion', 'Bulbasaur']
data.loc[(data['Name'].isin(my_pokemon))]
data.Attack.where(
    data.Attack > 49,
    49)  #filter the data greater than 49 and replace every other value to 49
data.dtypes
data.Name.cat.categories
data.Name.cat.ordered
data.Attack.max()
def create_bin(x):
    if x < 80:
        return 'low'
    elif x >= 80 and x < 120:
        return 'medium'
    else:
        return 'high'
data['attack_level'] = data.Attack.apply(create_bin)
data.head()
data['attack_level'] = data.attack_level.astype('category')
data.attack_level.cat.categories
data.attack_level.cat.codes
data.attack_level.cat.categories = ['H', 'L', 'M']
data.attack_level.cat.categories
data.head()
data.attack_level.cat.reorder_categories = ['H', 'M', 'L']
data.attack_level.cat.categories
number = np.arange(0, 5)
date_list = [
    "1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"
]
df = pd.DataFrame({'a': pd.to_datetime(date_list), 'b': number})
df
df.a.dt.day
df.a.dt.year
data.head()
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(
    data,
    id_vars='Name',
    value_vars=['Attack', 'Defense'],
    var_name='skill-type',
    value_name='value')
melted.head()
melted.pivot(index='Name', columns='skill-type', values='value').head()
data.assign(Nick=lambda x: 2 * x.Attack + 2 * x.Defense).head()
data['Nick'] = data.apply(lambda x: x.Attack + x.Defense, axis=1)
data.head()
data['random'] = np.random.randint(0, 100, data.shape[0])
data.head()
data1 = data.head()
data2 = data.tail()
data1.append(data2)
pd.concat([data1, data2], axis=0, ignore_index=True)  #stack by rows
data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat(
    [data1, data2], axis=1)  # axis = 0 : stack by columns
conc_data_col
data.set_index('Type 1', inplace=True)
data.head()
data1 = data.set_index(["Type 2", "Legendary"])
data.head(4)
data.groupby('Type 1').mean()  #it skipped categorical columns
data.groupby(level=0).max()  #aggregating by index: Type1
data1.groupby(level=1).mean()
data1.groupby(level=1).agg({'Attack': lambda x: sum(x) / len(x)})
data1.reset_index(inplace=True)
data1.head()
np.random.choice(['Ash', 'Brock', 'TeamKat'], 5)
pokemon_owner = pd.DataFrame({
    '#':
    data['#'],
    'Owner':
    np.random.choice(['Ash', 'Brock', 'TeamKat'], data.shape[0])
})
pokemon_owner.head()
pd.merge(data, pokemon_owner, how='left', on='#').head()
data.Attack.apply(lambda x: 2 * x).head()
data.drop(['Name'], axis=1)
data.rename(columns={'Name': 'Pokemon_name', 'Type 2': 'Type2'}).head()
data.rename(columns=str.lower).head()
data.values
# data.to_csv('../output/xyz.csv', index=False)
# data.to_feather('../outxyz')
data = pd.read_csv('../input/pokemon.csv')
data.head()
group = data.groupby(['Type 1', 'Type 2'])
group
group.last()
data.head()
data['Type 2'].groupby(data['Type 1']).count()
group = data.groupby(['Type 1', 'Type 2'])
group.get_group(('Water', 'Dark'))
group.aggregate(np.sum)
group.aggregate(np.sum).reset_index()
group.aggregate([np.sum, np.mean]).reset_index()
group.size()
group.describe()
