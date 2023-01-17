# Standard imports

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import plotly

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

import seaborn as sns

sns.set_style('whitegrid')

init_notebook_mode(connected=True)

# The data for the poll will be obtained from HuffPost Pollster.



# Grab data from web

import requests

# Work with csv file

from io import StringIO



# URL = 'https://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'



# Using request to get information in text form.

# source = requests.get(URL).text 



# Use StringIO to avoid IO error with pandas.

# poll_data = StringIO(source)

# Creating the dataframe

# data_df = pd.read_csv(poll_data)

data_df = pd.read_csv("../input/2012-election-obama-vs-romney/2012-general-election-romney-vs-obama.csv")

# Lets see what it looks like.

data_df.head()

# Lets see df info.

data_df.info()

# We see that most pollster's are non-affiliate, but are still somewhat dem leaning.

sns.countplot('Affiliation', data=data_df)

# Strong show of likely and registered voters

sns.countplot('Affiliation', data=data_df, hue='Population')

# Get voter averages

avg = pd.DataFrame(data_df.mean())

# Drop irrelevant columns

avg.drop(['Number of Observations', 'Question Text', 'Question Iteration'], axis=0, inplace=True)

# Get voter standard dev.

std = pd.DataFrame(data_df.std())

# Drop irrelevant columns

std.drop(['Number of Observations', 'Question Text', 'Question Iteration'], axis=0, inplace=True)

# Average sentiment of all polls

avg.plot(yerr=std, kind='bar', legend=False)

# Here we concatenate both avg and std datafames

poll_avg = pd.concat([avg, std], axis=1)

poll_avg.columns = ['Avg', 'Std']

poll_avg

# Timeseries

data_df.plot(x='End Date', y=['Obama', 'Romney', 'Undecided'], linestyle='', marker='o')

from datetime import datetime



# Here we find the difference in poll numbers. A positive percentage favors Obama,

# while a negative percentage favors Romney.

data_df['Difference'] = (data_df['Obama'] - data_df['Romney']) / 100



data_df.head()

# Take mean of grouped (start date) polls.

data_df = data_df.groupby(['Start Date'], as_index=False).mean()

data_df.head()

'''

Looks like Obama was leading in many of the polls through the election season. Romney received two

big spikes between late 2011 and early 2012. 

'''

# Time series of poll differences.

data_df.plot(x='Start Date', y='Difference', figsize=(12,4), marker='o', linestyle='-', color='green')

# Find row indexes of Start Date in October.

row_in = 0

xlimit = []



for date in data_df['Start Date']:

    if date[0:7] == '2012-10':

        xlimit.append(row_in)

        row_in += 1

    else:

        row_in += 1



print(f'Min: {min(xlimit)}', f'Max: {max(xlimit)}' )

# Time series of poll differences in the month of October.

data_df.plot(x='Start Date', y='Difference', figsize=(12,4), marker='o', linestyle='-', color='green', xlim=(325, 352))

# Debate on Oct. 3rd.

plt.axvline(x=325+2, linewidth=4, color='grey')

# Debate on Oct. 11th.

plt.axvline(x=325+10, linewidth=4, color='grey')

# Debate on Oct. 22nd.

plt.axvline(x=325+21, linewidth=4, color='grey')







donor_df = pd.read_csv("../input/2012-election-obama-vs-romney/Election_Donor_Data.csv")

# Lets see what it looks like.

donor_df.head()

# Lets see df info.

donor_df.info()

# Most common donation amounts

donor_df['contb_receipt_amt'].value_counts()

# Average donation amount

donation_mean = donor_df['contb_receipt_amt'].mean()

print(f'Average donation to candidates was ${donation_mean:0.2f}.')



# Standard deviation of donations

donation_std = donor_df['contb_receipt_amt'].std()

print(f'Standard deviation of donations to candidates was ${donation_std:0.2f}.')

top_donor = donor_df['contb_receipt_amt'].copy()

top_donor.sort_values()

# FCC records refund amounts in contb_receipt_amt.

# We will drop all refund amount and see how the std is affected.



top_donor = top_donor[top_donor > 0].sort_values()

top_donor.value_counts().head(10)

# Create a histogram to view peak donation amounts.



common_don = top_donor[top_donor < 2500]

common_don.hist(bins=100)

# List all candidates

candidates = donor_df['cand_nm'].unique()

candidates

party_dict = {

    'Bachmann, Michelle' : 'Republican',

    'Cain, Herman' : 'Republican',

    'Gingrich, Newt' : 'Republican',

    'Huntsman, Jon' : 'Republican',

    'Johnson, Gary Earl' : 'Republican',

    'McCotter, Thaddeus G' : 'Republican',

    'Obama, Barack' : 'Democrat',

    'Paul, Ron' : 'Republican',

    'Pawlenty, Timothy' : 'Republican',

    'Perry, Rick' : 'Republican',

    "Roemer, Charles E. 'Buddy' III" : 'Republican',

    'Romney, Mitt' : 'Republican',

    'Santorum, Rick' : 'Republican',

}



donor_df['party'] = donor_df.cand_nm.map(party_dict)



donor_df = donor_df[donor_df.contb_receipt_amt > 0]

donor_df.head()



def plot_bar(df, xaxis_title, yaxis_title, chart_title):



    color_dict_cand = {

    'Bachmann, Michelle' : 'crimson',

    'Cain, Herman' : 'crimson',

    'Gingrich, Newt' : 'crimson',

    'Huntsman, Jon' : 'crimson',

    'Johnson, Gary Earl' : 'crimson',

    'McCotter, Thaddeus G' : 'crimson',

    'Obama, Barack' : 'Blue',

    'Paul, Ron' : 'crimson',

    'Pawlenty, Timothy' : 'crimson',

    'Perry, Rick' : 'crimson',

    "Roemer, Charles E. 'Buddy' III" : 'crimson',

    'Romney, Mitt' : 'crimson',

    'Santorum, Rick' : 'crimson',

    }



    color_dict_party = {

    'Republican' : 'crimson',

    'Democrat' : 'blue',

    }



    if len(df.index) > 2:

        color = df.index.map(color_dict_cand)

    else:

        color = df.index.map(color_dict_party)



    trace = go.Bar(x=df.index, y=df, marker_color=color)

    layout = go.Layout(title=chart_title, 

                        xaxis={'title':xaxis_title},

                        yaxis={'title':yaxis_title})



    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig, filename=f'{df}')

# Count number of donations to each candidate

donor_df.groupby(['cand_nm'])['contb_receipt_amt'].count().sort_values(ascending=False)

candidate_donation_count = donor_df.groupby(['cand_nm'])['contb_receipt_amt'].count().sort_values(ascending=False)

i = 0

for count in candidate_donation_count:

    print(f'The candidate {candidate_donation_count.index[i]} received {count:.0f} total donations.')

    print()

    i += 1





xaxis = 'Candidates'

yaxis = 'Donation Count'

title = 'Candidate Donation count'

plot_bar(candidate_donation_count, xaxis, yaxis, title)

# Count number of donations to party

donor_df.groupby(['party'])['contb_receipt_amt'].count().sort_values(ascending=False)

party_donation_count = donor_df.groupby(['party'])['contb_receipt_amt'].count().sort_values(ascending=False)

i = 0

for count in party_donation_count:

    print(f'The {party_donation_count.index[i]} party received {count:.0f} total donations.')

    print()

    i += 1



xaxis = 'Party'

yaxis = 'Donation Count'

title = 'Party Donation count'

plot_bar(party_donation_count, xaxis, yaxis, title)

# Total donated to each candidate

donor_df.groupby(['cand_nm'])['contb_receipt_amt'].sum().sort_values(ascending=False)

candidate_donations = donor_df.groupby(['cand_nm'])['contb_receipt_amt'].sum().sort_values(ascending=False)

i = 0

for donation in candidate_donations:

    print(f'The candidate {candidate_donations.index[i]} raised a total of ${donation:.0f}.')

    print()

    i += 1



xaxis = 'Candidates'

yaxis = 'Donation Amount'

title = 'Candidate Donation Amounts'

plot_bar(candidate_donations, xaxis, yaxis, title)

# Total donated to party

donor_df.groupby(['party'])['contb_receipt_amt'].sum().sort_values(ascending=False)

candidate_party_donations = donor_df.groupby(['party'])['contb_receipt_amt'].sum().sort_values(ascending=False)

i = 0

for donation in candidate_party_donations:

    print(f'The {candidate_party_donations.index[i]} party raised a total of ${donation:.0f}.')

    print()

    i += 1



xaxis = 'Party'

yaxis = 'Donation Amount'

title = 'Party Donation Amounts'

plot_bar(candidate_party_donations, xaxis, yaxis, title)

# Average donated to each candidate

donor_df.groupby(['cand_nm'])['contb_receipt_amt'].mean().sort_values(ascending=False)



candidate_avg_donations = donor_df.groupby(['cand_nm'])['contb_receipt_amt'].mean().sort_values(ascending=False)

i = 0

for donation in candidate_avg_donations:

    print(f'The candidate {candidate_avg_donations.index[i]} had an average donation of ${donation:.0f}.')

    print()

    i += 1



xaxis = 'Candidates'

yaxis = 'Average Donation'

title = 'Candidate Avg Donation Amounts'

plot_bar(candidate_avg_donations, xaxis, yaxis, title)

# Average donated to party

donor_df.groupby(['party'])['contb_receipt_amt'].mean().sort_values(ascending=False)



party_avg_donations = donor_df.groupby(['party'])['contb_receipt_amt'].mean().sort_values(ascending=False)

i = 0

for donation in party_avg_donations:

    print(f'The {party_avg_donations.index[i]} party had an average donation of ${donation:.0f}.')

    print()

    i += 1



xaxis = 'Party'

yaxis = 'Average Donation'

title = 'Party DAvg onation Amounts'

plot_bar(party_avg_donations, xaxis, yaxis, title)



occupation_df = donor_df.pivot_table('contb_receipt_amt', 

                                    index='contbr_occupation', 

                                    columns='party', 

                                    aggfunc='sum')





print(f'There are over {occupation_df.shape[0]} reported occupations that have donated in the 2012 election season.')

occupation_df = occupation_df[occupation_df.sum(1) > 1000000]

occupation_df.drop(['INFORMATION REQUESTED PER BEST EFFORTS', 'INFORMATION REQUESTED'], axis=0, inplace=True)

occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

occupation_df.drop(['C.E.O.'], axis=0, inplace=True)



print(f'There were {occupation_df.shape[0]} occupations that as a group have donated at least $1,000,000 in the 2012 election season.')



occupation_df.plot(kind='barh', figsize=(10,12), cmap='seismic')
























