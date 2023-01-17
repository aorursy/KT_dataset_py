# import libraries

import pandas as pd 
series = pd.Series([1,2,3,4,5])

print(series[2:3])



series_1 = pd.Series([1,2,3,4], index = ['a', 'b', 'c', 'd'])

print(series_1)

print(series_1['a'])

print(series_1['d'])



print(series_1.index)

print(series_1.values)
# filter in python list 

l = [1,2,3,4,5]



def f_odd(x):

    return x % 2 == 1



l1 = filter(f_odd, l)

for x in l1:

    print(x)



# series_3 = pd.Series([1,2,3,4,5,6,7,'a'])

# series_3

# series_3[series_3 > 4]

# series[series % 2 == 1]

series[series.apply(f_odd)] * 100

    

# series[series % 2 == 1]


df = pd.DataFrame({

'country': ['Kazakhstan', 'Russia', 'Belarus', 'Ukraine', 'UK'],



'population': [17.04, 143.5, 9.5, 45.5, 45.5],



'square': [2724902, 17125191, 207600, 603628, 1]}, 

    

    

    index=['1', '2', '3', '4', '5'])

df



# x = df.iloc[[0, 0]]



df[['country']]

df[['country', 'square']]
# Locate by index name

# print(df.loc['1'])



# iloc: index locate

df.iloc[0]



df.loc[['1','2','4'], ['country', 'square']]



df.loc['1':'3', :]
df[df > 207600]

df[df['square'] >= 17125191]

df[df['country'].str.contains('u')]

df[df['square'] % 2 == 0]
df['square_2'] = df['square'] / 2

new_df = df.rename(columns={'country': 'quoc gia'})

new_df
df.sort_values(by='square', ascending=False)



df.sort_values(by=['population', 'square'], ascending=False)


csv_df = pd.read_csv('../input/nxqdautomobile/Automobile_data.csv', sep=',')

csv_df



csv_df.head(5)

csv_df.tail(5)

csv_df.sample(5)

# csv_df.info()
xs = csv_df[['company', 'price']]

xs = xs[csv_df['price'] == csv_df['price'].max()]

xs
x = csv_df.sort_values(by = 'price', ascending = False)

x.head(1)[['company', 'price']]