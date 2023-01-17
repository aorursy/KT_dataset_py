import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import os

INPUT_DIR = os.path.join("..", "input")

for dirname, _, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))

COMPETITION = "nlp-getting-started"

DATA_DIR = os.path.join(INPUT_DIR, COMPETITION)

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

TEST_FILE = os.path.join(DATA_DIR, "test.csv")

SAMPLE_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_FILE, TEST_FILE, SAMPLE_FILE
!head ../input/nlp-getting-started/sample_submission.csv
!head ../input/nlp-getting-started/train.csv
! wc -l ../input/nlp-getting-started/*.csv
train_df = pd.read_csv(TRAIN_FILE)

train_df.dtypes
train_df.keyword = train_df.keyword.str.strip()

train_df.location = train_df.location.str.strip()
train_df.shape
train_df.target.value_counts().plot.bar()

plt.title("Target Frequency", fontdict = {'fontsize' : 20})
total = len(train_df.keyword)

has_val = train_df.keyword.count()

print("{} of {} ({:.1f}%) of records have keyword values".format(has_val, total, 100 * has_val / total))

print("{} unique keywords".format(len(train_df.keyword.unique())))
kw30 = train_df.keyword.value_counts()[np.arange(30)]



# get positive target count for each keyword

pos_count = [train_df[train_df.keyword == kw].target.sum() for kw in kw30.keys()]

pos_series = pd.Series(pos_count, index = kw30.keys())



# get negative target count for each keyword

neg_count = [len(train_df[(train_df.keyword == kw) & (train_df.target == 0)]) for kw in kw30.keys()]

neg_series = pd.Series(neg_count, index = kw30.keys())



# stacked bar chart

plt.figure(figsize=(5, 10))

plt.title("Top 30 Keywords", fontdict = {'fontsize' : 20})



plt.barh(pos_series.keys(), pos_series.values, label="Disaster")

plt.barh(neg_series.keys(), neg_series.values, left = pos_series.values, label="Normal")

plt.legend()

plt.show()
total = len(train_df.location)

has_val = train_df.location.count()

print("{} of {} ({:.1f}%) of records have location values".format(has_val, total, 100 * has_val / total))

print("{} unique location values".format(len(train_df.location.unique())))
# get locations with 10 or more tweets

loc_freq = train_df.location.value_counts()[np.arange(30)]

#loc_freq = loc_freq[loc_freq >= 10]



# get positive target count for each keyword

pos_count = [train_df[train_df.location == loc].target.sum() for loc in loc_freq.keys()]

pos_series = pd.Series(pos_count, index = loc_freq.keys())



# get negative target count for each keyword

neg_count = [len(train_df[(train_df.location == loc) & (train_df.target == 0)]) for loc in loc_freq.keys()]

neg_series = pd.Series(neg_count, index = loc_freq.keys())



# stacked bar chart

plt.figure(figsize=(5, 10))

plt.title("Top 30 Locations", fontdict = {'fontsize' : 20})



plt.barh(pos_series.keys(), pos_series.values, label="Disaster")

plt.barh(neg_series.keys(), neg_series.values, left = pos_series.values, label="Normal")

plt.legend()

plt.show()

train_df.location.where(train_df.location.str.startswith('U')).unique()
train_df.location.where(train_df.location.isin(('US', 'USA', 'U.S.', 'U.S.A', 'America', 'United States', 'United States of America'))).unique()
top_combos = train_df.groupby(['location', 'keyword']).target.count().sort_values(ascending=False)[np.arange(50)]
ratio = [train_df[(train_df.location == loc) & (train_df.keyword == kw)].target.mean() for loc, kw in top_combos.keys()]

ratio_series = pd.Series(ratio, index = top_combos.keys())



combos_df = pd.DataFrame({'location': [loc for loc, kw in ratio_series.keys()], 

                          'keyword': [kw for loc, kw in ratio_series.keys()],

                          'ratio': ratio_series.values})

combos_df.sort_values(by = ['keyword', 'location'], inplace=True)

combos_df.head()
locations = combos_df.location.unique()

locations.sort()



keywords = combos_df.keyword.unique()

keywords.sort()



len(locations), len(keywords)
plt_data = np.zeros(shape=(len(locations), len(keywords)))



for idx, row in combos_df.iterrows():

    plt_data[np.where(locations == row.location), np.where(keywords == row.keyword)] = row.ratio

plt_data
fig, ax = plt.subplots(figsize=(20, 20))



im = ax.imshow(plt_data)



ax.set_xticks(np.arange(len(keywords)))

ax.set_yticks(np.arange(len(locations)))



ax.set_xticklabels(keywords)

ax.set_yticklabels(locations)



plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(locations)):

    for j in range(len(keywords)):

        text = ax.text(j, i, "{:.2f}".format(plt_data[i, j]),

                       ha="center", va="center", color="w")



ax.set_title("Ratio of Positive Target By Location and Keyword", fontdict = {'fontsize' : 20})

plt.show()