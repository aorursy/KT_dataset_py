# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sqlalchemy import create_engine



from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize


kiva = pd.read_csv('/kaggle/input/undata-country-profiles/kiva_country_profile_variables.csv')

countries = pd.read_csv('/kaggle/input/undata-country-profiles/country_profile_variables.csv')





# display(kiva.head())

# display(countries.head())





# Split Columns

engine = create_engine('sqlite://', echo=False)



countries.to_sql('countries', con=engine)







table = 'countries'



def split_column (old_col, new_col_1, new_col_2, table, engine, debug=False):

    

    if debug:

        query = """

            SELECT country, 

                    [{1}],

                    SUBSTR([{1}], 0,           INSTR([{1}], '/')   ) AS [{2}],

                    SUBSTR([{1}], INSTR([{1}],              '/')+1 ) AS [{3}]

            FROM {0}

        """.format(table, 

                   old_col, new_col_1, new_col_2)

    else:

        query = """

            SELECT  SUBSTR([{1}], 0,           INSTR([{1}], '/')   ) AS [{2}],

                    SUBSTR([{1}], INSTR([{1}],              '/')+1 ) AS [{3}]

            FROM {0}

        """.format(table, 

                   old_col, new_col_1, new_col_2)



    return pd.read_sql(query, engine)





composite_columns = []

new_df = pd.DataFrame()



debug_mode = False



#16 - Labour force participation (female/male pop. %)                       : 55.7/68.1

old_col = 'Labour force participation (female/male pop. %)'

new_col_1 = 'female_labour'

new_col_2 = 'male_labour'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)





#27 - Life expectancy at birth (females/males, years)                       : 81.2/76.5

old_col = 'Life expectancy at birth (females/males, years)'

new_col_1 = 'female_life'

new_col_2 = 'male_life'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)





#28 - Population age distribution (0-14 / 60+ years, %)                     : 18.9/21.5

old_col = 'Population age distribution (0-14 / 60+ years, %)'

new_col_1 = 'population_14'

new_col_2 = 'population_60'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)





# 29 - International migrant stock (000/% of total pop.)                     : 46627.1/14.5

old_col = 'International migrant stock (000/% of total pop.)'

new_col_1 = 'migrant_stock'

new_col_2 = 'migrant_stock_total'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)



# 35 - Education: Primary gross enrol. ratio (f/m per 100 pop.)              : 100.0/100.3

old_col = 'Education: Primary gross enrol. ratio (f/m per 100 pop.)'

new_col_1 = 'edu_1st_female'

new_col_2 = 'edu_1st_male'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)



# 36 - Education: Secondary gross enrol. ratio (f/m per 100 pop.)            : 98.5/96.7

old_col = 'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'

new_col_1 = 'edu_2nd_female'

new_col_2 = 'edu_2nd_male'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)



# 37 - Education: Tertiary gross enrol. ratio (f/m per 100 pop.)             : 99.6/72.8

old_col = 'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)'

new_col_1 = 'edu_3rd_female'

new_col_2 = 'edu_3rd_male'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)





# 43 - Forested area (% of land area)                                        : 5254.3/16.2

old_col = 'Forested area (% of land area)'

new_col_1 = 'frosted_area'

new_col_2 = 'land_area'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)



# 46 - Energy supply per capita (Gigajoules)                                 : 99.4/98.2

old_col = 'Energy supply per capita (Gigajoules)'

new_col_1 = 'energy_supply_per_capita'

new_col_2 = 'energy_supply_Gigajoules'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)   

composite_columns.append(old_col)



# 47 - Pop. using improved drinking water (urban/rural, %)                   : 100.0/100.0

old_col = 'Pop. using improved drinking water (urban/rural, %)'

new_col_1 = 'water_urban'

new_col_2 = 'water_rural'

new_df = pd.concat([new_df, split_column(old_col, new_col_1, new_col_2, table, engine, debug=debug_mode)], axis=1)

composite_columns.append(old_col)





new_df = new_df.fillna(-99)

new_df
for i, (name, value) in enumerate(zip(countries.columns, countries.iloc[217, :])):

    print('{} - {:70s}: {}'.format(i, name, value))
cleaned_data = countries



cleaned_data = cleaned_data.drop(columns=composite_columns)

cleaned_data = pd.concat([cleaned_data, new_df], axis=1)

# cleaned_data = cleaned_data.fillna(-99)



replacement = {'~0': 0,

              '-~0.0': 0,

               '~0.0': 0,

              '...': -99,

              '': -99}



cleaned_data = cleaned_data.replace(replacement)



# Data Cleaning

# col = 'Surface area (km2)'

# target = '~0'

# cleaned_data[col] = cleaned_data[col].replace(target, 0)

cleaned_data
cleaned_data.columns
# Normalize before performing K-means

information = cleaned_data.drop(columns=['country', 'Region'])



normalized_info = pd.DataFrame(normalize(information), columns=information.columns)

n_groups = 10





kmeans = KMeans(n_clusters=n_groups, random_state=0).fit(normalized_info)





groups = pd.DataFrame([cleaned_data['country'], kmeans.labels_]).T

groups.columns = ['country', 'group']





for i in range(n_groups):

    display(groups[groups['group'] ==i])
# A close look at Group 0



i = 1

display(groups[groups['group'] ==i].iloc[:50, 0])
display(groups[groups['group'] ==i].iloc[50:100, 0])