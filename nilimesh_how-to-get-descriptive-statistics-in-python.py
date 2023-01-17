# ========================================================================================

# Applied Data Science Recipes @ https://wacamlds.podia.com

# Western Australian Center for Applied Machine Learning and Data Science (WACAMLDS)

# ========================================================================================



print()

print(format('How to get descriptive statistics of a Pandas DataFrame','*^82'))    

import warnings

warnings.filterwarnings("ignore")



# load libraries

import pandas as pd



# Create dataframe

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 

            'age': [42, 52, 36, 24, 73], 

            'preTestScore': [4, 24, 31, 2, 3],

            'postTestScore': [25, 94, 57, 62, 70]}

df = pd.DataFrame(data, columns = ['name', 'age', 'preTestScore', 'postTestScore'])



print(); print(df)

print(); print(df.info())
# The sum of all the ages

print(); print(df['age'].sum())
# Mean preTestScore

print(); print(df['preTestScore'].mean())
# Cumulative sum of preTestScores, moving from the rows from the top

print(); print(df['preTestScore'].cumsum())
# Summary statistics on preTestScore

print(); print(df['preTestScore'].describe())
# Count the number of non-NA values

print(); print(df['preTestScore'].count())
# Minimum value of preTestScore

print(); print(df['preTestScore'].min())

# Maximum value of preTestScore

print(); print(df['preTestScore'].max())
# Median value of preTestScore

print(); print(df['preTestScore'].median())
# Sample variance of preTestScore values

print(); print(df['preTestScore'].var())
# Sample standard deviation of preTestScore values

print(); print(df['preTestScore'].std())
# Skewness of preTestScore values

print(); print(df['preTestScore'].skew())
# Kurtosis of preTestScore values

print(); print(df['preTestScore'].kurt())
# Correlation Matrix Of Values

print(); print(df.corr())
# Covariance Matrix Of Values

print(); print(df.cov())