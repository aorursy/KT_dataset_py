import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data=pd.read_csv('/kaggle/input/disney-movies/disney_movies.csv', parse_dates=['release_date'])
data.head()
# Sort data by the adjusted gross in descending order 

inflation_adjusted_gross_desc = data.sort_values(by='inflation_adjusted_gross', ascending=False) 



# Display the top 10 movies 

inflation_adjusted_gross_desc.head(10)
# Extract year from release_date and store it in a new column

data['release_year'] = pd.DatetimeIndex(data['release_date']).year



# Compute mean of adjusted gross per genre and per year

group = data.groupby(['genre','release_year']).mean()



# Convert the GroupBy object to a DataFrame

genre_yearly = group.reset_index()



# Inspect genre_yearly 

genre_yearly.head(10)
import seaborn as sns



# Plot the data  

sns.relplot(kind='line', x='release_year', y='inflation_adjusted_gross', hue='genre', data=genre_yearly)