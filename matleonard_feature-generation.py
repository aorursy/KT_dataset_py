

%matplotlib inline



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from sklearn.preprocessing import LabelEncoder



ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched'])



# Drop live projects

ks = ks.query('state != "live"')



# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))



# Timestamp features

ks = ks.assign(hour=ks.launched.dt.hour,

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)



# Label encoding

cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()

encoded = ks[cat_features].apply(encoder.fit_transform)



data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']

baseline_data = ks[data_cols].join(encoded)
interactions = ks['category'] + "_" + ks['country']

print(interactions.head(5))
label_enc = LabelEncoder()

data_interaction = baseline_data.assign(category_country=label_enc.fit_transform(interactions))

data_interaction.head()
# First, create a Series with a timestamp index

launched = pd.Series(ks.index, index=ks.launched, name="count_7_days").sort_index()

launched.head(20)
count_7_days = launched.rolling('7d').count() - 1

print(count_7_days.head(20))



# Ignore records with broken launch dates

plt.plot(count_7_days[7:]);

plt.title("Number of projects launched over periods of 7 days");
count_7_days.index = launched.values

count_7_days = count_7_days.reindex(ks.index)
count_7_days.head(10)
baseline_data.join(count_7_days).head(10)
def time_since_last_project(series):

    # Return the time in hours

    return series.diff().dt.total_seconds() / 3600.



df = ks[['category', 'launched']].sort_values('launched')

timedeltas = df.groupby('category').transform(time_since_last_project)

timedeltas.head(20)
# Final time since last project

timedeltas = timedeltas.fillna(timedeltas.median()).reindex(baseline_data.index)

timedeltas.head(20)
plt.hist(ks.goal, range=(0, 100000), bins=50);

plt.title('Goal');
plt.hist(np.sqrt(ks.goal), range=(0, 400), bins=50);

plt.title('Sqrt(Goal)');
plt.hist(np.log(ks.goal), range=(0, 25), bins=50);

plt.title('Log(Goal)');