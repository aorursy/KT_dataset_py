# Load some useful packages and configure plotting options

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')

# plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Helvetica'

plt.rcParams['font.monospace'] = 'Consolas'

plt.rcParams['font.size'] = 14

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['legend.fontsize'] = 14

plt.rcParams['figure.titlesize'] = 16

plt.rcParams['lines.linewidth'] = 2



%matplotlib inline



# for auto-reloading external modules

%load_ext autoreload

%autoreload 2
INPUT_DIR = '../input/'

TRAIN_FILE = 'train.json'

TEST_FILE = 'test.json'





train_df = pd.read_json(INPUT_DIR + TRAIN_FILE)

test_df = pd.read_json(INPUT_DIR + TEST_FILE)

all_df = pd.concat((train_df, test_df), axis=0) # Can be useful to facet on train vs test

# Add a bool to facet train vs test sets

all_df['train'] = all_df['interest_level'].notnull()

all_df['test'] = all_df['interest_level'].isnull()



n_train = train_df.shape[0]

n_test = test_df.shape[0]

n_total = n_train + n_test

n_train_pct = (n_train / n_total) * 100.0

n_test_pct = (n_test / n_total) * 100.0



print('Train DF Shape: {}, %age: {:.1f}'.format(train_df.shape, n_train_pct))

print('Test  DF Shape: {}, %age: {:.1f}'.format(test_df.shape, n_test_pct))
def print_df_info(df, name):

    """

    Prints out more detailed DF info

    """

    print('\n{} Info:\n'.format(name))

    print(df.info())

    print('\n{} Null info by column:\n'.format(name))

    print(df.isnull().sum(axis=0))

    print('\n{} Statistical Description:\n'.format(name))

    print(df.describe())



print_df_info(train_df, 'Train')

print_df_info(test_df, 'Test')
g = sns.FacetGrid(all_df, col="test", sharex=True, sharey=False, size=5)

g = g.map(sns.distplot, "bathrooms")
g = sns.FacetGrid(all_df[all_df['bathrooms'] < 10], col="test", sharex=True, sharey=False, size=5)

g = g.map(sns.countplot, "bathrooms")
g = sns.FacetGrid(all_df, col="test", sharex=True, sharey=False, size=5)

g = g.map(sns.countplot, "bedrooms")
# Check for correlation between bedrooms and bathrooms

sns.jointplot(data=all_df[all_df['train']], x="bedrooms", y="bathrooms", kind='reg');
# How many unique building IDs are there compared to the listings?

building_df = all_df.groupby('building_id').size().reset_index(name="count")

building_df = building_df.sort_values("count", ascending=False).reset_index(drop=True)

building_df.head(10)
# First of all we need to convert the date and time - "2016-06-24 07:54:24"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

train_df['created'] = pd.to_datetime(train_df['created'], format=DATE_FORMAT)

test_df['created'] = pd.to_datetime(test_df['created'], format=DATE_FORMAT)

all_df['created'] = pd.to_datetime(all_df['created'], format=DATE_FORMAT)
g = sns.FacetGrid(all_df, col="train", sharex=True, sharey=True, size=5)

g = g.map(sns.distplot, "bathrooms")
# How does the train / test split depend on the creation date?

all_agg_df = all_df.copy()

all_agg_df = all_agg_df.set_index('created', drop=True)

all_agg_df = all_agg_df.groupby('train').resample('1D').size().transpose()



fig, ax = plt.subplots(1,1, figsize=(18,10))

ax = all_agg_df.plot.bar(ax=ax, stacked=True)

ax.set_xticklabels(all_agg_df.index.strftime('%a %b %d'))

ax.set_title('All listings creation date and train/test split');
# How does creation vary by Day of Week?

all_agg_df = all_df.copy()

all_agg_df['dayofweek'] = all_df['created'].dt.dayofweek

all_agg_df = all_agg_df.groupby('dayofweek').size()



all_agg_df.plot.bar(title="All row creation by day of week");
# How does creation vary by Hour?

all_agg_df = all_df.copy()

all_agg_df['hour'] = all_df['created'].dt.hour

all_agg_df = all_agg_df.groupby('hour').size()



all_agg_df.plot.bar(title="All row creation by hour of day", figsize=(10,6));
# Separate line plot for day vs hour of day creation

all_agg_df = all_df.copy()



all_agg_df = all_agg_df.reset_index()

all_agg_df['dayofweek'] = all_agg_df['created'].dt.weekday_name

all_agg_df['hour'] = all_agg_df['created'].dt.hour

all_agg_df = all_agg_df.groupby(['dayofweek', 'hour']).size().reset_index(name="count")

# all_agg_df = all_agg_df[['dayofweek', 'hour', 'checkouts']]

all_agg_df = all_agg_df.pivot_table(values='count', index='hour', columns='dayofweek')



all_agg_df = all_agg_df[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

day_palette = sns.color_palette("hls", 7) # Need to have 7 distinct colours



fig, ax = plt.subplots(1,1, figsize=(10,8))

all_agg_df.plot.line(ax=ax, linewidth=3, color=day_palette, title="Created by hour and day");
# Proportion of interest level by day-of-week

INT_ORDER = ['low', 'medium', 'high']



plot_df = train_df.copy()

plot_df['day'] = plot_df['created'].dt.dayofweek

plot_df = plot_df.groupby(["day", "interest_level"]).size().reset_index(name='count')

day_total = plot_df.groupby('day').sum()   

plot_df = pd.merge(plot_df, day_total, left_on="day", right_index=True, suffixes=["_value", "_day_total"])

plot_df['fraction'] = plot_df['count_value'] / plot_df['count_day_total']

plot_df = plot_df.pivot_table(index="day", columns="interest_level", values="fraction")



plot_df = plot_df[INT_ORDER]

plot_df.plot.bar(stacked=True, title="Proportion of interest level by day-of-week")

plt.legend(bbox_to_anchor=(1.1, 0.7, 1., .102), loc=3);
# Proportion of interest level by hour-of-day

INT_ORDER = ['low', 'medium', 'high']



plot_df = train_df.copy()

plot_df['hour'] = plot_df['created'].dt.hour

plot_df = plot_df.groupby(["hour", "interest_level"]).size().reset_index(name='count')

day_total = plot_df.groupby('hour').sum()   

plot_df = pd.merge(plot_df, day_total, left_on="hour", right_index=True, suffixes=["_value", "_day_total"])

plot_df['fraction'] = plot_df['count_value'] / plot_df['count_day_total']

plot_df = plot_df.pivot_table(index="hour", columns="interest_level", values="fraction")



plot_df = plot_df[INT_ORDER]

plot_df.plot.bar(stacked=True, figsize=(8,6),

                 title="Proportion of interest level by day-of-week")

plt.legend(bbox_to_anchor=(1.05, 0.8, 1., .102), loc=3);
plot_df = all_df.copy()

plot_df['len_description'] = plot_df['description'].apply(len)

# plot_df['len_description'].plot.hist(bins=40, figsize=(8,6), title="Description length");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.distplot, "len_description")
plot_df = plot_df[plot_df['len_description'] < 2500]

#plot_df['len_description'].plot.hist(bins=40, figsize=(8,6), title="Description length (outliers removed)");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.distplot, "len_description")
plot_df = all_df.copy()

plot_df['words'] = plot_df['description'].apply(str.split).apply(len)

#plot_df['words'].plot.hist(bins=40, figsize=(8,6), title="Description words");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.distplot, "words")
plot_df = plot_df[plot_df['words'] < 500]

#plot_df['words'].plot.hist(bins=40, figsize=(8,6), title="Description words");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.distplot, "words")
# Interest level by length of description

plot_df = train_df.copy()

plot_df['len_description'] = plot_df['description'].apply(len)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by number of characters in description")

sns.violinplot(data=plot_df, x="interest_level", y="len_description", order=INT_ORDER, ax=ax);
# Interest level by length of description

plot_df = train_df.copy()

plot_df['words'] = plot_df['description'].apply(str.split).apply(len)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by number of words in description")

sns.violinplot(data=plot_df, x="interest_level", y="words", order=INT_ORDER, ax=ax);
# Plot Histograms of display address words for train and test sites

plot_df = all_df.copy()

plot_df['words'] = plot_df['display_address'].apply(str.split).apply(len)

#plot_df['words'].plot.hist(bins=40, figsize=(8,6), title="Description words");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.countplot, "words")
# Remove outliers and re-plot

plot_df = plot_df[plot_df['words'] < 12]

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.countplot, "words")
# Interest level by length of display_address

plot_df = train_df.copy()

plot_df['words'] = plot_df['display_address'].apply(str.split).apply(len)

plot_df = plot_df[plot_df['words'] < 12]

plot_df.head()



fix, ax = plt.subplots(1,1,figsize=(8,6))

sns.violinplot(data=plot_df, x="interest_level", y="words", order=INT_ORDER)

ax.set_title("Interest level by number of words in display address");
# Plot Histograms of display address words for train and test sites

plot_df = all_df.copy()

#plot_df['words'].plot.hist(bins=40, figsize=(8,6), title="Description words");

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.countplot, "latitude")
# Interest level by length of description

plot_df = train_df.copy()

plot_df['len_disp_address'] = plot_df['display_address'].apply(len)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by length of display address")

sns.violinplot(data=plot_df, x="interest_level", y="len_disp_address", order=INT_ORDER, ax=ax);
# Interest level by length of description

plot_df = plot_df[plot_df['len_disp_address'] < 20]

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by length of display address")

sns.violinplot(data=plot_df, x="interest_level", y="len_disp_address", order=INT_ORDER, ax=ax);
# Interest level by length of description words

plot_df = train_df.copy()

plot_df['disp_addess_words'] = plot_df['display_address'].apply(str.split).apply(len)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by words in display address")

sns.violinplot(data=plot_df, x="interest_level", y="disp_addess_words", order=INT_ORDER, ax=ax);
# Interest level by length of description words

plot_df = plot_df[plot_df['disp_addess_words'] < 8]

plot_df['disp_addess_words'] = plot_df['display_address'].apply(str.split).apply(len)

fix, ax = plt.subplots(1,1,figsize=(8,6))

ax.set_title("Interest level by words in display address")

sns.violinplot(data=plot_df, x="interest_level", y="disp_addess_words", order=INT_ORDER, ax=ax);
plot_df = all_df.copy()

g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.countplot, "latitude")
g = sns.FacetGrid(plot_df, col="test", sharex=True, sharey=True, size=5)

g = g.map(sns.countplot, "longitude")
plot_df.plot.scatter(x="latitude", y="longitude", title="Scatter plot of latitude/longitude");

# Class balance on interest level?

plot_df = train_df.groupby('interest_level').size()

plot_df = plot_df[INT_ORDER]

plot_df.plot.bar(title="Interest level counts in training data")



total = plot_df.sum()

print('Percentage of data:\n{}'.format((plot_df / total).round(3) * 100.0))