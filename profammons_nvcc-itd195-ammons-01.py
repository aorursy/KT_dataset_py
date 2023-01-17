import sys

# !conda install --yes --prefix {sys.prefix} datascience

!{sys.executable} -m pip install datascience



from datascience import *

%matplotlib inline

import matplotlib.pyplot as plot

plots.style.use('fivethirtyeight')

import numpy as np
data = 'http://www2.census.gov/programs-surveys/popest/datasets/2010-2015/national/asrh/nc-est2015-agesex-res.csv'

census_table = Table.read_table(data)

census_table
partial_census_table = census_table.select('SEX','AGE','POPESTIMATE2010','POPESTIMATE2015')

population = partial_census_table.relabeled('POPESTIMATE2010', '2010').relabeled('POPESTIMATE2015', '2015')

population.show(15) # show 15 rows instead of the default 10
pop_diff = population.column('2015') - population.column('2010')



# add two columns to the table

us_census = population.with_columns(

    'Difference', pop_diff,

    'Percent Diff', pop_diff/population.column('2010')  # this is a calculated column

    )



us_census.set_format('Percent Diff', PercentFormatter) # make it pretty :)
us_census.sort('Difference', descending=True)
max(us_census.where('AGE', are.below(999)).column('AGE'))
age_2010 = us_census.where('SEX', are.equal_to(0)).where('AGE', are.between(91,101)).drop('SEX').drop('2015').drop('Difference').drop('Percent Diff')

age_2010
# Display a horizontal bar chart of populations by age

age_2010.barh('AGE','2010')
# Generate a scatter plot with points and connecting lines (o-) -- this uses matplotlib instead of the built-in Table.plot method

plot.xlabel("Age")

plot.ylabel("Population")

plot.title("Populations by Age in 2010")

plot.plot(age_2010.column('AGE'), age_2010.column('2010'), '-o', color='blue')
# set the base URL for where the data comes from

path_data = 'https://raw.githubusercontent.com/data-8/textbook/gh-pages/data/'



# read the CSV file data into a table object called 'actors'

actors = Table.read_table(path_data + 'actors.csv')
# table_name.plot(x,y)
