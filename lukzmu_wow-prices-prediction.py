# Import libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()
# Read the data and preview sample rows

wow = pd.read_csv(

    '../input/world-of-warcraft-auction-house-data/tsm_data.csv',

    index_col='id',

    parse_dates=['created_at'],

    dtype={

        'item_name': 'category',

        'item_subclass': 'category',

    },

)

display(wow.sample(5, random_state=42))

wow.info()
# Filter out empty rows

wow = wow[wow['item_num_auctions'] > 0]



# Remove irrelevant columns

wow.drop(

    columns=[

        'item_vendor_buy',

        'item_vendor_sell',

        'item_market_value',

    ],

    inplace=True

)



# Update copper to gold

wow['item_min_buyout'] = wow['item_min_buyout'] / 10000



# Remove outliers

bad_ids = [1884, 1903, 1922, 1941, 1960]

wow.drop(index=bad_ids, inplace=True)



# Add days since raid release

ETERNAL_PALACE_RELEASE = pd.Timestamp('2019.07.10')

NYALOTHA_RELEASE = pd.Timestamp('2020.01.21')

def calculate_days_since_new_raid(date):

    ep_release = ETERNAL_PALACE_RELEASE

    ny_release = NYALOTHA_RELEASE

    if date < ny_release:

        return (date - ep_release).days

    else:

        return (date - ny_release).days



wow['days_after_new_raid'] = wow.apply(

    lambda row: calculate_days_since_new_raid(row['created_at']),

    axis=1,

)
# Preview dataframe after cleaning

display(wow.sample(5, random_state=42))

wow.info()
# Preview the dataset statistical numerics

wow.describe()
# Preview the timeseries of the dataset

wow['created_at'].describe()
# How many different items and subclasses are there?

unique_items = wow['item_name'].unique().tolist()

unique_subclass = wow['item_subclass'].unique().tolist()

f'There are {len(unique_items)} unique items in {len(unique_subclass)} subclasses.'
# List the items and their classes

wow.drop_duplicates(subset=['item_name'])[['item_name', 'item_subclass']].set_index('item_subclass')
# Preview correlations

sns.heatmap(wow.corr())
# Exclude feast from other due to big single item value

wow_feast = wow[wow['item_name'] == 'Famine Evaluator And Snack Table']

wow_other = wow[wow['item_name'] != 'Famine Evaluator And Snack Table']
# Check mean prices over the period for subclasses

wow_daily = wow_other.groupby([wow_other['created_at'].dt.date, 'item_subclass']).mean().reset_index()

wow_daily_feast = wow_feast.groupby([wow_feast['created_at'].dt.date]).mean().reset_index()



display('Mean values based on subclass:')

display(wow_daily.head())

display('Mean feast values based on subclass:')

display(wow_daily_feast.head())
# Daily mean prices per item subclass

plt.figure(figsize=(10,5))

ax = sns.lineplot(

    x='created_at',

    y='item_min_buyout',

    hue='item_subclass',

    data=wow_daily

)

ax.set_xlim(wow_daily['created_at'].min(), wow_daily['created_at'].max())

ax.set_xlabel('Dates')

ax.set_ylabel('Price in gold per item')

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

)

plt.axvline(NYALOTHA_RELEASE, color='black', linestyle='--', label='Ny\'alotha')

plt.show()
# The same but for Feasts

plt.figure(figsize=(10,5))

ax = sns.lineplot(

    x='created_at',

    y='item_min_buyout',

    data=wow_daily_feast

)

ax.set_xlim(wow_daily_feast['created_at'].min(), wow_daily_feast['created_at'].max())

ax.set_xlabel('Dates')

ax.set_ylabel('Price in gold per Feast')

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light',

),

plt.axvline(NYALOTHA_RELEASE, color='black', linestyle='--', label='Ny\'alotha')

plt.show()
# Check mean prices for days of week

wow_weekday = wow_other.groupby([wow_other['created_at'].dt.dayofweek, 'item_subclass']).mean().reset_index()

wow_weekday_feast = wow_feast.groupby(wow_feast['created_at'].dt.dayofweek).mean().reset_index()



display('Mean values based on subclass:')

display(wow_weekday.head())

display('Mean feast values based on subclass:')

display(wow_weekday_feast.head())
# Weekday mean prices per item subclass

plt.figure(figsize=(10,5))

days = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ax = sns.lineplot(

    x='created_at',

    y='item_min_buyout',

    hue='item_subclass',

    data=wow_weekday,

)

ax.set_xlabel('Day of week')

ax.set_ylabel('Price in gold per item')

ax.set(xticklabels=days)

plt.show()
# Weekday Feast mean prices per item subclass

plt.figure(figsize=(10,5))

days = ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ax = sns.lineplot(

    x='created_at',

    y='item_min_buyout',

    data=wow_weekday_feast,

)

ax.set_xlabel('Day of week')

ax.set_ylabel('Price in gold per Feast')

ax.set(xticklabels=days)

plt.show()
import torch

import torch.nn as nn 

import torch.autograd as autograd 

from torch.autograd import Variable

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
features = ['item_name', 'item_subclass', 'item_num_auctions', 'days_after_new_raid']

wow_features = wow_other[features]



# One Hot encoding

wow_features = pd.get_dummies(

    wow_features,

    columns=['item_name', 'item_subclass'],

)



# Scaling continuous values

wow_features[['item_num_auctions']] = preprocessing.scale(wow_features[['item_num_auctions']])

wow_features[['days_after_new_raid']] = preprocessing.scale(wow_features[['days_after_new_raid']])



# Display features

display(wow_features.columns)

display(wow_features[['item_num_auctions', 'days_after_new_raid']].head())



# Create and display target

wow_target = wow_other[['item_min_buyout']]

display(wow_target.head())
X_train, x_test, Y_train, y_test = train_test_split(

    wow_features,

    wow_target,

    test_size=0.2,

    random_state=42,

)
X_train_tr = torch.tensor(X_train.values, dtype=torch.float)

x_test_tr = torch.tensor(x_test.values, dtype=torch.float)

Y_train_tr = torch.tensor(Y_train.values, dtype=torch.float)

y_test_tr = torch.tensor(y_test.values, dtype=torch.float)



# Display sizes

display('X train size:', X_train_tr.shape)

display('Y train size:', Y_train_tr.shape)
input_size = X_train_tr.shape[1]

output_size = Y_train_tr.shape[1]

hidden_layers = 100

loss_function = torch.nn.MSELoss()

learning_rate = 0.0001
model = torch.nn.Sequential(

    torch.nn.Linear(input_size, hidden_layers),

    torch.nn.Sigmoid(),

    torch.nn.Linear(hidden_layers, output_size),

)
for i in range(10000):

    y_pred = model(X_train_tr)

    loss = loss_function(y_pred, Y_train_tr)

    

    if i % 1000 == 0:

        print(i, loss.item())

    

    model.zero_grad()

    loss.backward()

    

    with torch.no_grad():

        for param in model.parameters():

            param -= learning_rate * param.grad
sample = x_test.iloc[1410]

display(sample)



# Convert to tensor

sample_tr = torch.tensor(sample.values, dtype=torch.float)

display(sample_tr)
# Do predictions

y_pred = model(sample_tr)

print(f'Predicted price of item is: {int(y_pred.item())}')

print(f'Actual price of item is: {int(y_test.iloc[1410])}')
# Predict prices for entire dataset and show on graph

y_pred_tr = model(x_test_tr)

y_pred = y_pred_tr.detach().numpy()



plt.scatter(y_pred, y_test.values, s=1)

plt.xlabel("Actual Price")

plt.ylabel("Predicted price")



plt.title("Predicted prices vs Actual prices")

plt.show()