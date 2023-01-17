import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 100
color = sns.color_palette()
%matplotlib inline
FOLDER_ROOT = './..'
FOLDER_INPUT = FOLDER_ROOT + '/input'
FOLDER_OUTPUT = FOLDER_ROOT + '/output'
print(check_output(["ls", FOLDER_INPUT]).decode("utf8"))
countries_df = pd.read_csv(FOLDER_INPUT + '/countries.csv')
countries_df.head()
wid_df = pd.read_csv(FOLDER_INPUT + '/wid.csv')
wid_df.head()
variables_df = pd.read_csv(FOLDER_INPUT + '/wid_variables.csv', encoding='ISO-8859-1')
variables_df.head()
# Variable name contains `population`
population = variables_df['Variable Name'].str.contains("population", case=False)

# Variable name contains `all ages`
all_ages = variables_df['Variable Name'].str.contains('all ages', case=False)

# Variable name contains `individuals`
individuals = variables_df['Variable Name'].str.contains('individuals', case=False)
pd.set_option('display.max_colwidth', -1)

# Filter the available variables
result_df = variables_df[population & all_ages & individuals]

# Let's see the important columns
result_df[['Variable Code', 'Variable Name', 'Variable Description']]
my_var = 'npopul999i'
variables_df[variables_df['Variable Code'] == my_var].T
fr_population_df = wid_df[wid_df.country == 'FR'][['country', 'year', 'perc', my_var]]
fr_population_df.head()
plt.plot(fr_population_df['year'], fr_population_df[my_var] / 1000000)
plt.title("France population till 2016")
plt.xlabel("Years")
plt.ylabel("Population in million")
plt.show()