import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
# Create a Series

list_a = [1, 2, 3, 4, 5, 6, 7]

# series_a = pd.Series(None)

# series_a
list_b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# series_b = pd.Series(None)

# series_b
list_c = [1, 'bjdas', [8,9], {1,'s', 'e', 't'}, dict, print, float(22*0.14)]

# series_c = pd.Series(None)

# series_c
# Access the first element in the series

# series_a.iloc[None]
# Access the last element in the series

# series_a.iloc[None]
# Access the third to fifth elements in the series

# series_a.iloc[None]
series_d = pd.Series(list_a, index=list_b)

series_d
# series_d.loc[None]
selected_index = ['a', 'd', 'g']

# series_d.loc[None]
d = {'col1': [1, 2, 3, 4], 'col2': [3, 4, 10, 'a']}

# df = pd.DataFrame(None)
# df.iloc[None]
# df[None]
path = "../input/load_data/see_ess_vee.csv"

df = pd.read_csv(path)



# We can view the first 5 rows using df.head()



df.head()
df = pd.read_csv('../input/parks-data/parks.csv', index_col=['Park Code'])

df.head()
parks = ['HALE', 'HAVO', 'HOSP', 'ISRO', 'JOTR', 'KATM', 'KEFJ', 'KOVA', 'LACL','YOSE']

selected_df = df.loc[parks]

selected_df
selected_df['State']
mask = selected_df['State']=='HI'

selected_df[mask]
# selected_df[(selected_df['State']==None) &

#             (selected_df['Acres']>None)]
missing_data = {'col1': [1, 2, None, 4], 'col2': [None, 4, 10, 'a']}

missing_df = pd.DataFrame(missing_data)

missing_df
missing_df[missing_df['col1'].isnull()]
missing_df[~missing_df['col1'].isnull()]
# missing_df.rename(columns={None:None}, inplace=True)

# missing_df
# missing_df.drop(None, axis=1, inplace=True)

# missing_df
# missing_df.drop(None, axis=0, inplace=True)

# missing_df
d = {'col1': [1, 2, 3, 4], 'col2': [3, 4, 10, 1]}

df = pd.DataFrame(d)

df['minus'] = d['col1']-df['col2']

df
df['divide'] = d['col1']/df['col2']

df
df = pd.read_csv('../input/parks-data/parks.csv', index_col=['Park Code'])

df.sort_values(by='State', ascending=False, inplace=True)

df.head(10)
# Which state has the most national parks?
df.describe()
# df.groupby(None).agg(None)[None]
df = pd.DataFrame(data={'preds': [1, 0, 0, 0, 1, 0, 1, 1, 1],

                        'ground_truth': [1, 0, 1, 0, 1, 0, 0, 1, 1]})



confusion_matrix =pd.crosstab(df['ground_truth'], df['preds'], rownames=['ground_truth'], colnames=['Preds'])

print (confusion_matrix)
path = "../input/load_data/see_ess_vee.csv"

df = pd.read_csv(path)
# df.plot(x=None, y=None,kind=None)
df.plot(x='a', y='b',kind='scatter', color='k', title='Random Scatter')
datetime_example = pd.to_datetime('2019-08-01 12:40')

datetime_example
# We can extract hour/week/day from these timestamps using the following

print(datetime_example.hour)

print(datetime_example.weekday())

print(datetime_example.weekday_name)
df = pd.read_csv('../input/stock-time-series-20050101-to-20171231/all_stocks_2017-01-01_to_2018-01-01.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Date'].iloc[420]
# x is the row (Think of it as the 'i' in --- for i in iterable_object)

df['weekday_day'] = df['Date'].apply(lambda x: x.weekday_name) 
df.head()
# Function takes the value of that row and returns a value after processing

def part_of_week(row_entry):

    day_of_week = row_entry.weekday()

    if day_of_week in [0,1]:

        return 'early week'

    elif day_of_week == 2:

        return 'hump day'

    if day_of_week in [3,4]:

        return 'late week'



# Apply to the column you wishh to process and access each row using `x`

# Think of this technique as a for loop going over each row in a column

df['part_of_week'] = df['Date'].apply(lambda x: part_of_week(x))

        
df.head()
d1 = {'Customer_id':pd.Series([1,2,3,4,5,6]),

  'Product':pd.Series(['Oven','Oven','Oven','Television','Television','Television'])}

df1 = pd.DataFrame(d1)

 

d2 = {'Customer_id':pd.Series([2,4,6]),

    'State':pd.Series(['California','California','Texas'])}

df2 = pd.DataFrame(d2)
df1
df2
df3 = pd.merge(df1, df2, on='Customer_id', how='left')

df3
df1 = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

df1
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

df1.append(df2)
df_concat = pd.concat([df1, df2], axis=0)

df_concat
df_concat.loc[1]
df_concat = pd.concat([df1, df2], axis=0, ignore_index=True)

df_concat
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],

        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],

        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}



labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 

                               'Budapest_PaRis', 'Brussels_londOn'],

              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],

              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],

                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 

                               '12. Air France', '"Swiss Air"']})

df
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})



df = pd.DataFrame({'a': [None, None, None, 2, 3, 5],

                   'b': [1, 2 ,3, None, None, None]})