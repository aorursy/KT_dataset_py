# Data

import numpy as np

import pandas as pd



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))
csv_path = '../input/outfile.csv'

chess_data = pd.read_csv(csv_path)

chess_data.head()
chess_data.tail()
chess_data.columns
chess_data.shape
null_values = chess_data.isnull().sum()

print('Null Values:')

print(null_values.sort_values(ascending=False))
black_filter = chess_data['black'] == 'sammyteahan'

white_filter = chess_data['white'] == 'sammyteahan'



white_games = len(chess_data[white_filter])

black_games = len(chess_data[black_filter])



print('Games as white: {}'.format(white_games))

print('Games as black: {}'.format(black_games))
##

# map date column (str) to date type

#

chess_data.date = pd.to_datetime(chess_data.date,

                                 format='%Y.%m.%d')

chess_data.head()
##

# Add Wins/Losses/elo to dataframe for easier counting and grouping

#

def apply_win(row):

    return 1 if 'sammyteahan won' in row['result'] else 0



def apply_loss(row):

    return 0 if 'sammyteahan won' in row['result'] else 1



def apply_elo(row):

    return row['white_elo'] if 'sammyteahan' in row['white'] else row['black_elo']



chess_data['win'] = chess_data.apply(lambda row: apply_win(row), axis=1)

chess_data['loss'] = chess_data.apply(lambda row: apply_loss(row), axis=1)

chess_data['my_elo'] = chess_data.apply(lambda row: apply_elo(row), axis=1)



##

# Random sampling to see our new data

#

chess_data.take(np.random.permutation(len(chess_data))[:10])
##

# Grouping wins and losses

#



# Extract month to new column to make grouping easier

chess_data['month'] = pd.DatetimeIndex(chess_data['date']).month



wins = chess_data['win'].groupby(chess_data['month']).sum()

losses = chess_data['loss'].groupby(chess_data['month']).sum()
##

# More win/loss data

#

win_filter = chess_data['win'] == 1

loss_filter = chess_data['loss'] == 1



win_count = len(chess_data[win_filter])

loss_count = len(chess_data[loss_filter])



print('Total wins: {}'.format(win_count))

print('Total losses: {}'.format(loss_count))

print('Win percentage: {:.2f}%'.format(((win_count / (win_count + loss_count)) * 100)))
##

# Monthly aggregations

#

monthly_data = chess_data.date.dt.month

month_totals = monthly_data.value_counts() # group by month

month_labels = month_totals.index # x vals

month_counts = month_totals.get_values() # y vals

months = ['Jan', 'Feb', 'Mar', 'Apr',

          'May', 'Jun', 'Jul', 'Aug',

          'Sep', 'Oct', 'Nov', 'Dec']



def monthly_wins_and_losses():

    with sns.axes_style('darkgrid'):

        fig, ax = plt.subplots(figsize=(20, 10))

        sns.barplot(x=month_labels, y=month_counts,

                    color='#0984e3', label='Total', ax=ax)

        sns.barplot(x=wins.index, y=wins.get_values(),

                    color='#81ecec', label='Wins', ax=ax)

        ax.set_xlabel('')

        ax.set_ylabel('# Games')

        ax.set_title('Games totals per month')

        ax.legend(ncol=2, loc="upper right", frameon=True)

        ax.set_xticklabels(months)



monthly_wins_and_losses()
##

# Daily aggregations

#

daily_data = chess_data.date.dt.dayofweek

daily_totals = daily_data.value_counts() # grouped by day

daily_labels = daily_totals.index # x vals

daily_counts = daily_totals.get_values() # y vals

labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



# add day to dataframe for easy grouping

chess_data['day'] = pd.DatetimeIndex(chess_data['date']).dayofweek

daily_wins = chess_data['win'].groupby(chess_data['day']).sum()

daily_losses = chess_data['loss'].groupby(chess_data['day']).sum()



def plot_daily_wins_and_losses():

    with sns.axes_style('darkgrid'):

        fig, ax = plt.subplots(figsize=(20, 10))

        sns.barplot(x=daily_labels, y=daily_counts,

                    color='#0984e3', label='Total', ax=ax)

        sns.barplot(x=daily_wins.index, y=daily_wins.get_values(),

                    color='#81ecec', label='Wins', ax=ax)

        ax.set_xlabel('')

        ax.set_ylabel('# Games')

        ax.set_title('Games total by day')

        ax.set_xticklabels(labels)

        ax.legend(ncol=2, loc='upper right', frameon=True)



# just making sure our data frame is still good

print(daily_data.value_counts().sort_index().sum())

plot_daily_wins_and_losses()
##

# Hourly aggregations

#



# need to map end time to datetime field

# _and_ change timezone

# chess_data.end_time = pd.to_datetime(chess_data.end_time, format=)



chess_data
# drop unused 'round' column

chess_data = chess_data.drop(['round'], axis=1)

chess_data.head()
def plot_line_chart():

    with sns.axes_style('dark'):

        fix, ax = plt.subplots(figsize=(20, 10))

        sns.lineplot(x=chess_data['date'], y=chess_data['my_elo'])



plot_line_chart()