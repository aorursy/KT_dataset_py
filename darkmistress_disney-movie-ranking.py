# Import pandas library

# ... YOUR CODE FOR TASK 1 ...

import pandas as pd



# Read the file into gross

gross = pd.read_csv('../input/datasets1/disney_movies_total_gross.csv', parse_dates = ['release_date'])



# Print out gross

gross.head()
# Sort data by the adjusted gross in descending order 

# ... YOUR CODE FOR TASK 2 ...

inflation_adjusted_gross_desc = gross.sort_values(by='inflation_adjusted_gross', ascending=False)



# Display the top 10 movies 

# ... YOUR CODE FOR TASK 2 ...

inflation_adjusted_gross_desc.head(10)
 # Extract year from release_date and store it in a new column

gross['release_year'] = pd.DatetimeIndex(gross['release_date']).year



# Compute mean of adjusted gross per genre and per year

group = gross.groupby(['genre', 'release_year']).mean()



# Convert the GroupBy object to a DataFrame

genre_yearly = group.reset_index()



# Inspect genre_yearly 

genre_yearly.head(10)
# Import seaborn library

# ... YOUR CODE FOR TASK 4 ...

import seaborn as sns



# Plot the data  

# ... YOUR CODE FOR TASK 4 ...

sns.relplot(kind='line', x='release_year', y='inflation_adjusted_gross', hue='genre', data=genre_yearly)
# Convert genre variable to dummy variables 

genre_dummies =  pd.get_dummies(data = gross['genre'], drop_first=True)



# Inspect genre_dummies

genre_dummies.head()
# Import LinearRegression

# ... YOUR CODE FOR TASK 6 ...

from sklearn.linear_model import LinearRegression



# Build a linear regression model

regr = LinearRegression()



# Fit regr to the dataset

# ... YOUR CODE FOR TASK 6 ...

regr.fit(genre_dummies, gross['inflation_adjusted_gross'])



# Get estimated intercept and coefficient values 

action =  regr.intercept_

adventure = regr.coef_[[0]][0]



# Inspect the estimated intercept and coefficient values 

print((action, adventure))
# Import a module

import numpy as np



# Create an array of indices to sample from 

inds = np.arange(len(gross['genre']))



# Initialize 500 replicate arrays

size = 500

bs_action_reps =  np.empty(size)

bs_adventure_reps =  np.empty(size)
# Generate replicates  

for i in range(size):

    

    # Resample the indices 

    bs_inds = np.random.choice(inds, size=len(inds))

        

    # Get the sampled genre and sampled adjusted gross

    bs_genre = gross['genre'][bs_inds]

    bs_gross = gross['inflation_adjusted_gross'][bs_inds]

    

    # Convert sampled genre to dummy variables

    bs_dummies = pd.get_dummies(bs_genre, drop_first=True)  

    

    # Build and fit a regression model 

    regr = LinearRegression().fit(bs_dummies, bs_gross)

    

    # Compute replicates of estimated intercept and coefficient

    bs_action_reps[i] = regr.intercept_

    bs_adventure_reps[i] = regr.coef_[[0]][0]
# Compute 95% confidence intervals for intercept and coefficient values

confidence_interval_action = np.percentile(bs_action_reps, [])

confidence_interval_adventure = np.percentile(bs_adventure_reps, [2.5, 97])

    

# Inspect the confidence intervals

print(confidence_interval_action)

print(confidence_interval_adventure)
# should Disney studios make more action and adventure movies? 

more_action_adventure_movies = True