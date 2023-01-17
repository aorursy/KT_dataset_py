import pandas as pd
import numpy as np
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
movies = pd.read_csv('http://bit.ly/imdbratings')
orders = pd.read_csv('http://bit.ly/chiporders', sep='\t')
orders['item_price'] = orders.item_price.str.replace('$', '').astype('float')
stocks = pd.read_csv('http://bit.ly/smallstocks', parse_dates=['Date'])
titanic = pd.read_csv('http://bit.ly/kaggletrain')
ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])
pd.__version__
pd.show_versions()
df = pd.DataFrame({'col one':[100, 200], 'col two':[300, 400]})
df
pd.DataFrame(np.random.rand(4, 8))
pd.DataFrame(np.random.rand(4, 8), columns=list('abcdefgh'))
df
df = df.rename({'col one':'col_one', 'col two':'col_two'}, axis='columns')
df.columns = ['col_one', 'col_two']
df.columns = df.columns.str.replace(' ', '_')
df
df.add_prefix('X_')
df.add_suffix('_Y')
drinks.head()
drinks.loc[::-1].head()
drinks.loc[::-1].reset_index(drop=True).head()
drinks.loc[:, ::-1].head()
drinks.dtypes
drinks.select_dtypes(include='number').head()
drinks.select_dtypes(include='object').head()
drinks.select_dtypes(include=['number', 'object', 'category', 'datetime']).head()
drinks.select_dtypes(exclude='number').head()
df = pd.DataFrame({'col_one':['1.1', '2.2', '3.3'],
                   'col_two':['4.4', '5.5', '6.6'],
                   'col_three':['7.7', '8.8', '-']})
df
df.dtypes
df.astype({'col_one':'float', 'col_two':'float'}).dtypes
pd.to_numeric(df.col_three, errors='coerce')
pd.to_numeric(df.col_three, errors='coerce').fillna(0)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
df
df.dtypes
drinks.info(memory_usage='deep')
cols = ['beer_servings', 'continent']
small_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols)
small_drinks.info(memory_usage='deep')
dtypes = {'continent':'category'}
smaller_drinks = pd.read_csv('http://bit.ly/drinksbycountry', usecols=cols, dtype=dtypes)
smaller_drinks.info(memory_usage='deep')
pd.read_csv('../input/dataset/stocks1.csv')
pd.read_csv('../input/dataset/stocks2.csv')
pd.read_csv('../input/dataset/stocks3.csv')
from glob import glob
stock_files = sorted(glob('../input/dataset/stocks*.csv'))
stock_files
pd.concat((pd.read_csv(file) for file in stock_files))
pd.concat((pd.read_csv(file) for file in stock_files), ignore_index=True)
pd.read_csv('../input/dataset/drinks1.csv').head()
pd.read_csv('../input/dataset/drinks2.csv').head()
drink_files = sorted(glob('../input/dataset/drinks*.csv'))
pd.concat((pd.read_csv(file) for file in drink_files), axis='columns').head()
len(movies)
movies_1 = movies.sample(frac=0.75, random_state=1234)
movies_2 = movies.drop(movies_1.index)
len(movies_1) + len(movies_2)
movies_1.index.sort_values()
movies_2.index.sort_values()
movies.head()
movies.genre.unique()
movies[(movies.genre == 'Action') |
       (movies.genre == 'Drama') |
       (movies.genre == 'Western')].head()
movies[movies.genre.isin(['Action', 'Drama', 'Western'])].head()
movies[~movies.genre.isin(['Action', 'Drama', 'Western'])].head()
counts = movies.genre.value_counts()
counts
counts.nlargest(3)
counts.nlargest(3).index
movies[movies.genre.isin(counts.nlargest(3).index)].head()
ufo.head()
ufo.isna().sum()
ufo.isna().mean()
ufo.dropna(axis='columns').head()
ufo.dropna(thresh=len(ufo)*0.9, axis='columns').head()
df = pd.DataFrame({'name':['John Arthur Doe', 'Jane Ann Smith'],
                   'location':['Los Angeles, CA', 'Washington, DC']})
df
df.name.str.split(' ', expand=True)
df[['first', 'middle', 'last']] = df.name.str.split(' ', expand=True)
df
df.location.str.split(', ', expand=True)
df['city'] = df.location.str.split(', ', expand=True)[0]
df
df = pd.DataFrame({'col_one':['a', 'b', 'c'], 'col_two':[[10, 40], [20, 50], [30, 60]]})
df
df_new = df.col_two.apply(pd.Series)
df_new
pd.concat([df, df_new], axis='columns')
orders.head(10)
orders[orders.order_id == 1].item_price.sum()
orders.groupby('order_id').item_price.sum().head()
orders.groupby('order_id').item_price.agg(['sum', 'count']).head()
orders.head(10)
orders.groupby('order_id').item_price.sum().head()
len(orders.groupby('order_id').item_price.sum())
len(orders.item_price)
total_price = orders.groupby('order_id').item_price.transform('sum')
len(total_price)
orders['total_price'] = total_price
orders.head(10)
orders['percent_of_total'] = orders.item_price / orders.total_price
orders.head(10)
titanic.head()
titanic.describe()
titanic.describe().loc['min':'max']
titanic.describe().loc['min':'max', 'Pclass':'Parch']
titanic.Survived.mean()
titanic.groupby('Sex').Survived.mean()
titanic.groupby(['Sex', 'Pclass']).Survived.mean()
titanic.groupby(['Sex', 'Pclass']).Survived.mean().unstack()
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean',
                    margins=True)
titanic.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='count',
                    margins=True)
titanic.Age.head(10)
pd.cut(titanic.Age, bins=[0, 18, 25, 99], labels=['child', 'young adult', 'adult']).head(10)
titanic.head()
pd.set_option('display.float_format', '{:.2f}'.format)
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
 .background_gradient(subset='Volume', cmap='Blues')
)
(stocks.style.format(format_dict)
 .hide_index()
 .bar('Volume', color='lightblue', align='zero')
 .set_caption('Stock Prices from October 2016')
)
import pandas_profiling
pandas_profiling.ProfileReport(titanic)
