import numpy as np

import pandas as pd



from sklearn.model_selection import KFold, cross_val_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

sns.set(palette='Set1')

%matplotlib inline
# Read in the data

df = pd.read_csv('/kaggle/input/police-pedestrian-stops-and-vehicle-stops/police_pedestrian_stops_and_vehicle_stops.csv')
# Take another look

df.head()
# Check for NaN values

df.isnull().sum()
df.columns = [col.lower() for col in df.columns]
# That's much better!

df.columns
# Take a quick glance at call_disposition as a value count

df.call_disposition.value_counts().head(20)
# Create the arrest column based on call_disposition to establish our label in 1s and 0s

# Checking a lowercase version ensures capitalization won't throw us off

df['arrest_made'] = df.call_disposition.apply(lambda x: 1 if 'arrest' in x.lower() else 0)
# Let's see what percentage of stops resulted in an arrest

df.arrest_made.mean()*100
# Let's break down the time_phonepickup column into more specific time chunks



# First we should convert the date as a string into a datetime object

df.time_phonepickup = pd.to_datetime(df.time_phonepickup)



# Then we can use datetime attributes to find hour (0-23), day of the week (0 is Monday, Sunday is 6), month (1-12), and year

df['hour'] = df.time_phonepickup.apply(lambda x: x.hour)

df['day_of_week'] = df.time_phonepickup.apply(lambda x: x.weekday()) # isoweekday() returns Monday starting as 1 and Sunday as 6

df['month'] = df.time_phonepickup.apply(lambda x: x.month)

df['year'] = df.time_phonepickup.apply(lambda x: x.year)
# Let's take a look at the precinct

df.precinct_id.value_counts()
df = df[df['precinct_id'] != 'None']
# That got me thinking, what if there are 'None' values in other columns. To find out, let's loop through each column and check.

for column in df.columns:

    if any(df[column] == 'None'):

        print(column)

        print(len(df[df[column]=='None']))
# It comes out to less than .2% of our dataframe records, so I think it's safe to drop them. We can do this in one line by returning the df only if all values

# Within each row are not 'None'

df = df[(df[df.columns] != 'None').all(axis=1)]
# Here's what we want to keep



to_keep = ['problem', 'arrest_made', 'hour', 'day_of_week', 'month', 'year', 'neighborhood_name', 'precinct_id']

new_df = df[to_keep]

"""

Here I tried label encoding as opposed to one hot, to no avail

new_df['neighborhood_name'] = df.neighborhood_name.astype('category').cat.codes

new_df['precinct_id'] = df.precinct_id.astype('category').cat.codes

"""

to_keep.pop(1)
# Now let's use pandas get_dummies function to one hot encode the categorical columns

encoded_df = pd.get_dummies(new_df, columns=to_keep, drop_first=True)
#test_df = encoded_df.sample(100000)



# We establish our features variable (X) and our label variable (y). Notice X is everything but 'arrest_made'.



kf = KFold(n_splits=3, random_state = 1)



X = encoded_df.drop('arrest_made', axis=1)

y = encoded_df.arrest_made

print(X.shape)

print(y.shape)
# # Now we instantiate our class with the number of neighbors we want to try. 

# knn = KNeighborsClassifier(n_neighbors=20)

# scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()

# print(scores)
### If we had time, we could loop over a range for n_neighbors to find the best fit



# k_range = (1,30)

# k_scores = []

# for k in k_range:

#     knn = KNeighborsClassifier(n_neighbors=k)

#     scoring = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()

#     k_scores.append(scoring)

# print(k_scores)
def get_accuracy(model):

    if model == KNeighborsClassifier:

        return cross_val_score(model(n_neighbors=20), X, y, cv=kf, scoring='accuracy').mean()

    else:

        return cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
for model in [LogisticRegression, RandomForestClassifier]:#, KNeighborsClassifier]:

    mean = get_accuracy(model())

    print("Model: {} - mean: {}".format(model.__name__, mean))