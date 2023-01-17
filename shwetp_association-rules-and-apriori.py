import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data = {'ID':[1,2,3,4,5,6],
       'Onion':[1,0,0,1,1,1],
       'Potato':[1,1,0,1,1,1],
       'Burger':[1,1,0,0,1,1],
       'Milk':[0,1,1,1,0,1],
       'Beer':[0,0,1,0,1,0]}
df = pd.DataFrame(data)
df = df[['ID', 'Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]]
df
frequent_itemsets = apriori(df[['Onion', 'Potato', 'Burger', 'Milk', 'Beer' ]], 
                            min_support=0.50, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules
rules [ (rules['lift'] >1.125)  & (rules['confidence']> 0.8)  ]
retail_shopping_basket = {'ID':[1,2,3,4,5,6],
                         'Basket':[['Beer', 'Diaper', 'Pretzels', 'Chips', 'Aspirin'],
                                   ['Diaper', 'Beer', 'Chips', 'Lotion', 'Juice', 'BabyFood', 'Milk'],
                                   ['Soda', 'Chips', 'Milk'],
                                   ['Soup', 'Beer', 'Diaper', 'Milk', 'IceCream'],
                                   ['Soda', 'Coffee', 'Milk', 'Bread'],
                                   ['Beer', 'Chips']
                                  ]
                         }
retail = pd.DataFrame(retail_shopping_basket)
retail = retail[['ID', 'Basket']]
pd.options.display.max_colwidth=100
retail
retail = retail.drop('Basket' ,1).join(retail.Basket.str.join(',').str.get_dummies(','))
retail
frequent_itemsets_2 = apriori(retail.drop('ID',1), use_colnames=True)
frequent_itemsets_2
association_rules(frequent_itemsets_2, metric='lift')
association_rules(frequent_itemsets_2)
movies = pd.read_csv('../input/movies.csv')
movies.head(10)
movies_ohe = movies.drop('genres',1).join(movies.genres.str.get_dummies())
pd.options.display.max_columns=100
movies_ohe.head()
stat1 = movies_ohe.drop(['title', 'movieId'],1).apply(pd.value_counts)
stat1.head()
stat1 = stat1.transpose().drop(0,1).sort_values(by=1, 
                                                ascending=False).rename(columns={1:'No. of movies'})
stat1.head()
stat2 = movies.join(movies.genres.str.split('|').reset_index().genres.str.len(), rsuffix='r').rename(
    columns={'genresr':'genre_count'})
stat2.head(10)
stat2 = stat2[stat2['genre_count']==1].drop('movieId',1).groupby('genres').sum().sort_values(
    by='genre_count', ascending=False)
stat2.head(10)
stat2.shape
stat = stat1.merge(stat2, how='left', left_index=True, right_index=True).fillna(0)
stat.genre_count=stat.genre_count.astype(int)
stat.rename(columns={'genre_count': 'No. of movies with only 1 genre'},inplace=True)
stat.head()
movies_ohe.set_index(['movieId','title'],inplace=True)
movies_ohe.head()
frequent_itemsets_movies = apriori(movies_ohe,use_colnames=True, min_support=0.025)
frequent_itemsets_movies
rules_movies =  association_rules(frequent_itemsets_movies, metric='lift', min_threshold=1.25)
rules_movies