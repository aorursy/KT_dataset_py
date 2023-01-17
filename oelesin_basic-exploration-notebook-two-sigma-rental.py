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
%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

color = sns.color_palette()
## Load JSON data with pandas in

train_df = pd.read_json('../input/train.json')

test_df = pd.read_json('../input/test.json')
## Basic information about the train and test sets

print("Train Rows : ", train_df.shape[0])

print("Test Rows : ", test_df.shape[0])
fig, ax = plt.subplots(1, figsize=(10, 7))



ax.set_ylabel("Number of Occurencies", fontsize=16) 

train_df.groupby(['interest_level']).size().sort_values(0, False).plot(kind='bar', ax=ax, rot=0, color=color[3]);

ax.set_xlabel('Interest Level', fontsize=16);
## view the interest levels by average price of houses

fig, ax = plt.subplots(1, figsize=(10, 7))



ax.set_ylabel("AVG Price of Property", fontsize=16) 

train_df.groupby(['interest_level'])['price'].mean().sort_values(0, False).plot(kind='bar', ax=ax, rot=0, color=color[3])

ax.set_xlabel('Interest Level', fontsize=16);
## explore interest levels by average number of bedrooms and bathrooms

fig, ax = plt.subplots(1, figsize=(10, 7))



ax.set_ylabel("AVG Number of Bedrooms & Bathrooms", fontsize=16) 

train_df.groupby(['interest_level'])['bedrooms', 'bathrooms'].mean().plot(kind='bar', ax=ax, rot=0)

ax.set_xlabel('Interest Level', fontsize=16);
## explore interest levels by average number of bedrooms and bathrooms

fig, axs = plt.subplots(1, 2, figsize=(12, 7))



axs[0].set_ylabel("Number of Occurencies", fontsize=16)

train_df.groupby(['bedrooms']).size().plot(kind='bar', ax=axs[0], rot=0, color=color[3])



axs[1].set_ylabel("Number of Occurencies", fontsize=16)

train_df.groupby(['bathrooms']).size().plot(kind='bar', ax=axs[1], rot=0, color=color[3])



axs[0].set_xlabel('Bedrooms', fontsize=16);

axs[1].set_xlabel('Bathrooms', fontsize=16);
## lets explore interest levels over time

## convert the created column into datetime series object

train_df['created'] = pd.to_datetime(train_df['created'])

train_df['month_created'] = train_df['created'].apply(lambda x: x.strftime('%B'))
plt.figure(figsize=(12,7))

sns.countplot(x='bedrooms', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurrences', fontsize=16)

plt.xlabel('Bedrooms', fontsize=16)

plt.show()
plt.figure(figsize=(12,7))

sns.countplot(x='bathrooms', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurrences', fontsize=16)

plt.xlabel('Bedrooms', fontsize=16)

plt.show()
def assign_time_of_day(date):

	"""

	This function computes and returns hour of the day

	Function arguments MUST be of type datetime.datetime

	"""

	if not date:

		return "unknown"

	hour_of_day = date.hour

	times_of_day = {

		'morning' : range(7, 12),

		'lunch'	: range(12, 14),

		'afternoon' : range(14, 18),

		'evening' : range(18, 24),

		'night': range(0, 7)

	}

	for k, v in times_of_day.items():

		if hour_of_day in v:

			 return k
train_df['hour_created'] = train_df['created'].apply(assign_time_of_day)
fig, ax = plt.subplots(1, figsize=(12, 7))



ax.set_ylabel("Number of Occurrencies", fontsize=16) 

train_df.groupby(['hour_created']).size().sort_values(0, False).plot(kind='bar', rot=0, ax=ax, color=color)

ax.set_xlabel('Hour Created', fontsize=16);
fig, ax = plt.subplots(1, figsize=(12, 7))



ax.set_ylabel("AVG Property Price", fontsize=16) 

train_df.groupby(['hour_created'])['price'].mean().sort_values(0, False).plot(kind='bar', rot=0, ax=ax, color=color)

ax.set_xlabel('Hour Created', fontsize=16);
train_df['num_of_photos'] = train_df.photos.apply(len)
plt.figure(figsize=(12,7))

sns.countplot(x='num_of_photos', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurrences', fontsize=16)

plt.xlabel('Bedrooms', fontsize=16)

plt.show()
train_df["num_features"] = train_df["features"].apply(len)

cnt_srs = train_df['num_features'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of features', fontsize=12)

plt.show()