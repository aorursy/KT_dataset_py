# Importing Libraries

import numpy as np

import pandas as pd

from pandas import Series,DataFrame



# For Visualization

import matplotlib as plt

import seaborn as sns

sns.set_style('whitegrid')

sns.set(font_scale=2)

% matplotlib inline



# Resize the size of plots

fig_size = plt.rcParams["figure.figsize"]

 

# Set figure width to 12 and height to 9

fig_size[0] = 14

fig_size[1] = 10

plt.rcParams["figure.figsize"] = fig_size
# Import Dataset

poll = pd.read_csv('../input/2016-general-election-trump-vs-clinton.csv')

poll.head()
# Drop various colums from data as we dont need them

poll = poll.drop(['Pollster URL','Source URL','Partisan','Question Text','Question Iteration'],axis=1)

poll.head()
# Glimpse of data

poll.info()
# Quick visualization overview of the affiliation for the polls

sns.factorplot('Affiliation',data=poll,kind='count',legend=True,color='orange',size=6)
sns.factorplot('Affiliation',data=poll,kind='count',legend=True,hue='Population',size=6,aspect=2,palette='dark')
# Averages of different candidates



average=pd.DataFrame(poll.mean())

average.drop('Number of Observations',inplace=True)

average
# Standard Deviation to get the error



std=pd.DataFrame(poll.std())

std.drop('Number of Observations',inplace=True)

std
average.plot(yerr=std,kind='bar',legend=False,color='seagreen',fontsize=20)
# Consider undecided factor



poll_avg=pd.concat([average,std],axis=1)

poll_avg.column=['Average','STD']

poll_avg
# Time series plot of favour of voters against time

poll.plot(x='End Date',y=['Trump','Clinton','Other','Undecided'],linestyle='',marker='s').legend(bbox_to_anchor=(1.5, 1))
# Plot out the difference between Tump and Clinton and how it changes as time moves along

from datetime import datetime



poll['Difference']=(poll.Trump-poll.Clinton)/100

poll.head()
# Visualize how this sentiment in difference changes over time



poll=poll.groupby('Start Date',as_index=False).mean()

poll.head()
# Plotting the difference in polls between Trump and Clinton

poll.plot('Start Date','Difference',figsize=(25,15),marker='s',color='red')
# Plot marker lines on the dates of the debates and see if there is any general insight to the poll results

# The debate dates were Sept 26th, Oct 9th and Oct 19th 0f 2016



row_in=0

xlimit=[]



for date in poll['Start Date']:

    if date[0:7] == '2016-09':

        xlimit.append(row_in)

        row_in +=1

    else:

        row_in +=1

        

print (min(xlimit))



row_in=0

xlimit=[]



for date in poll['Start Date']:

    if date[0:7] == '2016-10':

        xlimit.append(row_in)

        row_in +=1

    else:

        row_in +=1

        

print (max(xlimit))
poll.plot('Start Date','Difference',figsize=(25,15),marker='s',color='red',xlim=(209,262))

plt.pyplot.axvline(x=209+27, linewidth=4, color='grey')

plt.pyplot.axvline(x=209+40, linewidth=4, color='grey')

plt.pyplot.axvline(x=209+50, linewidth=4, color='grey')
# Import Donor data

donor=pd.read_csv('../input/Donor_Data.csv')

donor.head()
# Quick Overview

donor.info()
# Get a quick look at the various donation amounts

donor['contb_receipt_amt'].value_counts()
# Remove rows where amount is in negative

donor.drop(donor[donor.contb_receipt_amt<0].index, inplace=True)
donor['contb_receipt_amt'].value_counts()
# Mean and STD on donation amount



donor_mean=donor['contb_receipt_amt'].mean()

donor_std=donor['contb_receipt_amt'].std()



print("Average donation was: %0.2f with a standard deviation of: %0.2f" %(donor_mean,donor_std)) 
# Sort the data to get the top donations



top_donation=donor['contb_receipt_amt'].copy()

top_donation.sort_values(ascending=False, inplace=True)

top_donation.head(10)
top_donation.value_counts(sort=True).head(10)
# Create a series of donation limited to 2500

com_don=top_donation[top_donation<2500]

com_don.hist(bins=100)
# Grab candidate names from data

candidates=donor.cand_nm.unique()

candidates
# Dictionary of party affiliation

party_map = {'Rubio, Marco': 'Republican',

           'Santorum, Richard J.': 'Republican',

           'Perry, James R. (Rick)': 'Republican',

           'Carson, Benjamin S.': 'Republican',

           "Cruz, Rafael Edward 'Ted'": 'Republican',

           'Paul, Rand': 'Republican',

           'Clinton, Hillary Rodham': 'Democrat',}

           



# Now map the party with candidate

donor['Party'] = donor.cand_nm.map(party_map)
donor.head()
#  Quick look a the total amounts received by each candidate

donor.groupby('cand_nm')['contb_receipt_amt'].count()
# Total donation received by each candidate

donor.groupby('cand_nm')['contb_receipt_amt'].sum()
# To make it more readable

cand_amount=donor.groupby('cand_nm')['contb_receipt_amt'].sum()

i=0

for don in cand_amount:

    print("The candidate %s raised %.0f dollars" %(cand_amount.index[i],don))

    print("\n")

    i +=1
# Plot graph

cand_amount.plot(kind='bar',legend=True,logy=True, color='blue')
# Compare Democrat versus Republican donations

donor.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar',legend=True,logy=True,color='seagreen')
# Occupations of dononrs

occupation=donor.pivot_table('contb_receipt_amt',index='contbr_occupation',columns='Party', aggfunc='sum')

occupation
occupation = occupation[occupation.sum(1) > 1000000]
occupation

# Plot occupation

occupation.plot(kind='barh',figsize=(10,12),fontsize=12)
# Drop rows where there is no information

occupation.drop(['INFORMATION REQUESTED PER BEST EFFORTS','INFORMATION REQUESTED'],axis=0,inplace=True)
occupation.plot(kind='barh',figsize=(10,12),cmap='seismic')