import pandas as pd

import numpy as np

%pylab inline
def defdf():

    df = pd.DataFrame({'int_col' : [1, 2, 6, 8, -1], 

                   'float_col' : [0.1, 0.2, 0.2, 10.1, None], 

                   'str_col' : ['a', 'b', None, 'c', 'a']})

    return df



df = defdf()

df
df2 = pd.DataFrame({'str_col_2' : ['a','b'], 'int_col_2' : [1, 2]})

df2
plot_df = pd.DataFrame(np.random.randn(1000,2),columns=['x','y'])

plot_df['y'] = plot_df['y'].map(lambda x : x + 1)

plot_df.head()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# redefining the example objects



# series

population = pd.Series({'Germany': 81.3, 'Belgium': 11.3, 'France': 64.3, 

                        'United Kingdom': 64.9, 'Netherlands': 16.9})



# dataframe

data = {'country': ['Belgium', 'France', 'Germany', 'Netherlands', 'United Kingdom'],

        'population': [11.3, 64.3, 81.3, 16.9, 64.9],

        'area': [30510, 671308, 357050, 41526, 244820],

        'capital': ['Brussels', 'Paris', 'Berlin', 'Amsterdam', 'London']}

countries = pd.DataFrame(data)

countries
#Setting the index to the country names:





countries = countries.set_index('country')

countries
countries['area']


countries[['area', 'population']]
countries['France':'Netherlands']
countries.loc['Germany', 'area']
countries.loc['France':'Germany', ['area', 'population']]


countries.iloc[0:2,1:3]


countries2 = countries.copy()

countries2.loc['Belgium':'Germany', 'population'] = 10
countries2
countries['area'] > 100000


countries[countries['area'] > 100000]