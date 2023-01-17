# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import World University Ranking data set

educ_data_set = pd.read_csv("../input/world-university-rankings/cwurData.csv")
# Change all instances of 'USA' to 'United States'

educ_data_set.country[educ_data_set.country=='USA'] = 'United States'
# Display World University Ranking data set

educ_data_set
# Import and display World Happiness Index data set

happy_data_set = pd.read_csv("../input/world-happiness/2015.csv")

happy_data_set = happy_data_set.rename(columns={'Country': 'country_h', 'Region': 'region', 'Happiness Rank':'world_rank', 'Happiness Score':'score', 'Standard Error':'std_error', 'Economy (GDP per Capita)':'economy', 'Family':'family', 'Health (Life Expectancy)':'health', 'Freedom':'freedom', 'Trust (Government Corruption)':'trust', 'Generosity':'generosity', 'Dystopia Residual':'dystopia_residual'})

happy_data_set
# Compute for the average of the world ranking per country

average_educ_rank = pd.DataFrame((educ_data_set.groupby('country'))['world_rank'].mean())

average_educ_rank = average_educ_rank.sort_values(by=['world_rank'])



average_educ_rank

average_emp_rank = pd.DataFrame((educ_data_set.groupby('country'))['alumni_employment'].mean())

average_emp_rank = average_emp_rank.sort_values(by=['alumni_employment'])

average_emp_rank

# Compute Pearson's correlation for World University Ranking

correlation_table = educ_data_set.corr(method='pearson', min_periods=1)

correlation_table
import seaborn as sns

sns.heatmap(correlation_table, 

            xticklabels=correlation_table.columns.values,

            yticklabels=correlation_table.columns.values)
# Display World Happiness Index rankings

happy_rank = pd.DataFrame((happy_data_set['country_h']),(happy_data_set['world_rank']))

happy_rank
# NEED TO PROVE RELATIONSHIP BETWEEN UNI RANKS AND HAPPINESS INDEX

# How to combine the two?



happy_rank.join(average_educ_rank,how='left', on='country')