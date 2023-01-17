import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import os





%matplotlib inline
print(os.listdir('../input'))
df = pd.read_csv('../input/googleplaystore.csv')
df.info()
df.head()
df[df['Type'].isna()]
df.replace({'Type': None}, 'Free', inplace=True)
df[df['Content Rating'].isna()]
def shift_right(column):

    new_columns = [column[0]]

    counter = 1

    for i in range(1, len(column)):

        if i == 1:

            new_columns.append(None)

        else: 

            new_columns.append(column[counter])

            counter += 1

    return new_columns

        

    
x = df[df['Content Rating'].isna()].values
shifted = shift_right(x[0])
df[df['Content Rating'].isna()] = shifted
df[df['App'] == 'Life Made WI-Fi Touchscreen Photo Frame']
df['Category'].unique()
df.replace({'Category': None}, 'PHOTOGRAPHY', inplace=True)
df['Genres'].unique()
df.replace({'Genres': None}, 'Photography', inplace=True)
len(df[(df['Current Ver'].isna()) | (df['Android Ver'].isna())])
df.dropna(inplace=True)
df['Size'].unique()
def size_convert(size):

    

   return "".join(list(size)[:-1])

    

    
k = df[df['Size'].apply(lambda x : list(x)[-1] == 'k')]
k['Size'] = k['Size'].apply(size_convert)
k
k['Size'] = k['Size'].astype(float)
k['Size'] = k['Size'] / 1000
m = df[df['Size'].apply(lambda x : list(x)[-1] == 'M')]

m['Size'] = m['Size'].apply(lambda x : "".join(list(x)[:-1]))
m['Size'] = m['Size'].astype(float)
size_df = pd.concat([k, m ])
size_df['Size'] = size_df['Size'].astype(float)
sns.set_style('darkgrid')



plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)

sns.boxplot(y='Size', x='Category', data=size_df)
size_dict = size_df.groupby('Category').mean().to_dict()['Size']
vwd = df[df['Size'] == 'Varies with device']
vwd['new_size'] = vwd['Category'].map(size_dict)
vwd['Size'] = vwd['new_size']
vwd.drop('new_size', axis=1, inplace=True)
new_df = pd.concat([vwd, k, m])
new_df = new_df.reset_index().sort_values(by='index', ascending=True)
new_df.index = new_df['index']
new_df.head()
new_df.drop('index', axis=1, inplace=True)
new_df.head()
new_df['Size'] = new_df['Size'].round(2)
plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)



sns.boxplot(y='Size', x='Category', data=new_df)
new_df.info()
new_df['Installs'].unique()
def remove_mark (installs):

    numbers = list(installs)[:-1]

    

    while ',' in numbers:

        numbers.remove(',')

    return "".join(numbers)

    
new_df['Installs'] = new_df['Installs'].apply(remove_mark)
new_df['Installs'] = new_df['Installs'].astype(int)
order = new_df.groupby('Installs').count().sort_values(by='App',ascending=True).index.values



plt.xticks(rotation=90)



sns.countplot(x='Installs', data=new_df, order=order)
def remove_dollar_sign(price):

    if price == '0':

        return price

    else:

        return "".join(list(price[1:]))
new_df['Price'] = new_df['Price'].apply(remove_dollar_sign)
new_df['Price'] = new_df['Price'].astype(float)
new_df.info()
new_df['Reviews'] = new_df['Reviews'].astype(int)

new_df['Rating'] = new_df['Rating'].astype(float)
sns.pairplot(new_df[['Reviews', 'Size','Installs', 'Price', 'Rating']])
sns.heatmap(new_df.corr(), annot=True)
sns.countplot(x='Type', data=new_df)
sns.barplot(x='Type', y='Rating', data=new_df)
new_df.sort_values(by='Price',ascending=False)[new_df['Price'] > 200]
sns.scatterplot(y='Rating', x='Price', data=new_df, hue='Type')
new_df.head()
plt.figure(figsize=(12, 4))

sns.scatterplot(x='Installs', y='Rating', data=new_df)

plt.xlim(0, 1000000000)
sns.barplot(y='Rating', x='Content Rating', data=new_df)
new_df['Reviews per Install'] = new_df['Reviews'] / new_df['Installs']
sns.scatterplot(x='Reviews per Install', y='Rating', data=new_df)
new_df[new_df['Reviews per Install'] > 1].sort_values(by='Reviews per Install', ascending=False)
plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)





sns.boxplot(y='Rating', x='Category', data=new_df)


plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)





sns.barplot(y='Installs', x='Category', data=new_df.groupby('Category').sum().reset_index())
new_df['Last Updated'] = pd.to_datetime(new_df['Last Updated'], yearfirst=True )
plt.figure(figsize=(16, 4))

sns.lineplot(x='Last Updated', y='App', data=new_df.groupby('Last Updated').count().reset_index().sort_values(by='Last Updated', ascending=True))
plt.figure(figsize=(16, 4))

sns.lineplot(x='Last Updated', y='Rating', data=new_df.groupby('Last Updated').mean().reset_index().sort_values(by='Last Updated', ascending=True))
new_df.head()
new_df['Rounded Rating'] = new_df['Rating'].round()
sns.swarmplot(y='Rating', x='Type', hue='Content Rating', data=new_df[new_df['Category'] == 'GAME'])
new_df['Revenue'] = new_df['Installs'] * new_df['Price']


plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)



rev = new_df.groupby('Category').sum().sort_values(by='Revenue', ascending=False).reset_index()

sns.barplot(x='Category', y='Revenue', data=rev)
new_df.sort_values(by='Revenue', ascending=False).head(10)
new_df.drop(new_df[new_df.duplicated('App')].index.get_values(), inplace=True)
df_length = new_df.count()
new_df.index = np.arange(0, df_length[0])
new_df.replace({'Category': {1661: 'GAME'}}, inplace=True)
new_df[new_df['App'] == 'Minecraft']['Category'] = 'GAME'
new_df.loc[1661, 'Category'] = 'GAME'


plt.figure(figsize=(16, 4))

plt.xticks(rotation=90)



rev = new_df.groupby('Category').sum().sort_values(by='Revenue', ascending=False).reset_index()

sns.barplot(x='Category', y='Revenue', data=rev)


plt.figure(figsize=(16,4))

plt.xticks(rotation=90)



sns.violinplot(x='Category', y='Rating', data=new_df)
new_df.head()


plt.figure(figsize=(16,4))

plt.xticks(rotation=90)



sns.barplot(x='Category', y='Reviews', data=new_df.groupby('Category').sum().sort_values(by='Reviews', ascending=False).reset_index())