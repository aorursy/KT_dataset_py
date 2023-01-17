# Link to original notebook from Data School: https://nbviewer.jupyter.org/github/justmarkham/pandas-videos/blob/master/top_25_pandas_tricks.ipynb

# It's only my copy- I did it to play around with those tricks. Don't consider me as an author of those tips.



import numpy as np

import pandas as pd
drinks = pd.read_csv('http://bit.ly/drinksbycountry')

movies = pd.read_csv('http://bit.ly/imdbratings')

orders = pd.read_csv('http://bit.ly/chiporders', sep='\t')

orders['item_price'] = orders.item_price.str.replace('$', '').astype('float')

stocks = pd.read_csv('http://bit.ly/smallstocks', parse_dates=['Date'])

titanic = pd.read_csv('http://bit.ly/kaggletrain')

ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])
drinks.head()
movies.head()
orders.head()
stocks.head()
titanic.head()
ufo.head()
pd.__version__
pd.show_versions()
# Creating DataFrame out of dictionary- key->column, value->values of rows

df = pd.DataFrame({'col1':[123, 321], 'col2':[456, 654]})

df
# Creating DataFrame with ndarray, e.g. with random numbers:

df_rand = pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))

df_rand
df = df.rename({'col1':'col_1', 'col2':'col_2'}, axis='columns')

df
df.columns = ['col_one', 'col_two']

df
df.columns = df.columns.str.replace('_', '#')

df
df.add_prefix('X_')

df.add_suffix('Y_')
drinks.head()
drinks.iloc[::-1].head()
drinks.iloc[::-1].reset_index(drop=True).head()
drinks.iloc[:, ::-1].head()
drinks.dtypes
drinks.select_dtypes(include='number').dtypes # float, int, object, bool, category,datetime
drinks.select_dtypes(exclude='number').dtypes # float, int, object, bool, category,datetime
df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],

                   'col_two':['4.4', '5.5', '6.6'],

                   'col_three':['7.7', '8.8', '-']})

df
df.dtypes
df.astype({'col_one':'float', 'col_two':'float'}).dtypes # col_three has '-' so it's impossible to easily cast it with 'astype'

df.dtypes
pd.to_numeric(df.col_three, errors='coerce')
pd.to_numeric(df.col_three, errors='coerce').fillna(0)
print(df.dtypes)



df = df.apply(pd.to_numeric, errors='coerce').fillna(0)



print(df.dtypes)
drinks.info(memory_usage='deep')
cols = ['beer_servings', 'continent']

pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols).info(memory_usage='deep')
dtypes = {'continent':'category'}

pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols, dtype=dtypes).info(memory_usage='deep')
df1 = pd.DataFrame({'col1':[1, 2], 'col2':[3, 4]})

df2 = pd.DataFrame({'col1':[11, 22], 'col2':[33, 44]})

df3 = pd.DataFrame({'col1':[111, 222], 'col2':[333, 444]})



df_combined_rows = pd.concat([df1, df2, df3]).reset_index(drop=True)

df_combined_rows
df_combined_cols = pd.concat([df1, df2, df3], axis='columns')

df_combined_cols
# df = pd.read_clipboard()
print(len(movies))



movies_1 = movies.sample(frac=0.25, random_state=42)

movies_2 = movies.drop(movies_1.index)



print(len(movies_1)/len(movies))

print(len(movies_2)/len(movies))



movies_1.sort_index() # Sort DataFrame by index

movies_2.sort_index()



print(movies_2.index.sort_values())

print(movies_2.index.sort_values())
movies.head()
movies.genre.unique()
movies[(movies.genre == 'Crime') | (movies.genre == 'Horror') | (movies.genre == 'Adventure')].genre.unique()
movies[movies.genre.isin(['Crime', 'Horror', 'Adventure'])].genre.unique()
movies[~movies.genre.isin(['Crime', 'Horror', 'Adventure'])].genre.unique() # Excluding listed genres
counts = movies.genre.value_counts()

counts
counts.nlargest(3)
counts.nlargest(3).index
movies[movies.genre.isin(counts.nlargest(3).index)].head()
ufo.head()
ufo.isna().sum()
ufo.dropna(axis='columns').head() # drop columns with NaNs
ufo.dropna(thresh=len(ufo)*0.9, axis='columns').head() # Drop columns which have over 10% missing values
df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],

                   'location':['Los Angeles, CA', 'Washington, DC']})

df
df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)

df
df['city'] = df.location.str.split(',', expand=True)[0]

df
df = df.drop(columns=['name', 'location'], axis='columns') # drop unnecessary columns

df
df = pd.DataFrame({'col_1':[12, 13, 14], 'col_2':[[2,3], [4,5], [6,7]]})

df
df_new = df.col_2.apply(pd.Series)

df_new
df = pd.concat([df, df_new], axis='columns')

df = df.drop(columns=['col_2'], axis='columns')

df
orders.head()
orders[orders.order_id == 1].item_price.sum()
orders.groupby('order_id').item_price.sum().head()
orders.groupby('order_id').item_price.agg(['sum', 'count']).head()
orders.head()
print(len(orders.groupby('order_id').item_price.sum()))

print(len(orders.item_price))
orders['total_price'] = orders.groupby('order_id').item_price.transform('sum')



orders['percent_per_total'] = orders.total_price / orders.item_price

orders.head(10)
titanic.head()
titanic.describe()
titanic.describe().loc['min':'max', ['Survived', 'Pclass', 'Fare']]
titanic.Survived.mean()
titanic.groupby('Sex').Survived.mean()
titanic.groupby(['Sex', 'Pclass']).Survived.mean()
titanic.groupby(['Sex', 'Pclass']).Survived.mean().unstack()
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean', margins=True)
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='count', margins=True)
titanic.Age.head(10)
titanic['age_categorized'] = pd.cut(titanic.Age, bins=[0, 18, 25, 99], labels=['child', 'young adult', 'adult'])

titanic.head()
titanic.pivot_table(index='age_categorized', columns='Pclass', values='Survived', aggfunc='mean', margins=True)
titanic.head()
pd.set_option('display.float_format', '{:.1f}'.format)

titanic.head()
pd.reset_option('display.float_format')
stocks
format_dict = {'Date':'{:%m/%d/%y}', 'Close':'${:.2f}', 'Volume':'{:,}'}

stocks.style.format(format_dict)
(stocks.style.format(format_dict)

 .hide_index()

 .highlight_min('Close', color='red')

 .highlight_max('Close', color='lightgreen')

)
(stocks.style.format(format_dict)

 .hide_index()

 .background_gradient(subset='Volume', cmap='Greys')

)
(stocks.style.format(format_dict)

 .hide_index()

 .bar('Volume', color='gray', align='zero')

 .set_caption('Stock Prices from October 2016')

)
import pandas_profiling
pandas_profiling.ProfileReport(titanic)