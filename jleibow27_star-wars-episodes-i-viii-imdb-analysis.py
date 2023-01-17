# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns; sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read in csv

imdb = pd.read_csv('../input/movie_metadata_modificado3.csv')
# Print Head

print(imdb.head())



# Print Info

imdb.info()
# Find the values contained in the "Star Wars" group 

df = pd.DataFrame(imdb, columns = ['movie_title', 'director_name', 'title_year', 'budget', 'gross', 'imdb_score'])



sw = df[imdb['movie_title'].str.contains('Star Wars')]

sw_sorted = sw.sort_values(by=['title_year'])

sw_sorted
# Drop the titles with missing values

sw_drop = sw_sorted.dropna()

sw_drop
# Add 2 new rows: Episode VII and VIII, which were not adequately represented or present in our imdb dataset

VII_VIII = [pd.Series(['Star Wars: Episode VII - The Force Awakens?', 

                       'J.J. Abrams', 2015.0, 245000000.0, 936662225.0, 8.0], index=sw_drop.columns ),

            

            pd.Series(['Star Wars: Episode VIII - The Last Jedi?', 

                       'Rian Johnson', 2017.0, 317000000.0, 620181382.0, 7.1], 

                      index=sw_drop.columns )]



sw_full = sw_drop.append(VII_VIII, ignore_index=True)
#reset index

sw_index = sw_full.reset_index(drop=True)

sw_index
#remove unwanted ? at the end of the movie titles by creating a new column called movie_title_

sw_clean2 = sw_index.assign(movie_title_=sw_index['movie_title'].str.replace(r'?', ''))



#drop original movie_title column containing the movie titles with ?

sw_drop2 = sw_clean2.drop(['movie_title'], axis=1)

sw_drop2
#re-arrange columns so that movie_title_ is first

cols = sw_drop2.columns.tolist()

cols = cols[-1:] + cols[:-1]

cols

#rename cols to sw_arranged and display as a df

sw_arranged = sw_drop2[cols]

sw_arranged
#remove director name and title year to leave only the columns used for analysis

sw_budget_gross = sw_arranged[['movie_title_','budget','gross', 'imdb_score']]

sw_budget_gross_plot = sw_budget_gross.sort_index(ascending=False)

sw_budget_gross_plot
# Import df for a horizontal bar plot showing the budgets of all 8 Star Wars movies

ax = sw_budget_gross_plot.plot(kind='barh', x='movie_title_', y='budget', color='gold')



# Despine

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)



# Set Title

ax.set_title('Star Wars Movies - Budget (in order of release date)')



# Set x-axis label

ax.set_xlabel("Budget (in millions)", labelpad=20, weight='bold', size=12)



# Set y-axis label

ax.set_ylabel("Movie Title", labelpad=20, weight='bold', size=12)



# Set xticks for budget

start, end = ax.get_xlim()

ax.xaxis.set_ticks(np.arange(start, end, 350000000))

ax.set_xticks([0, 50000000, 100000000, 150000000, 200000000, 250000000, 300000000, 350000000])

ax.set_xticklabels(['0', '50M', '100M', '150M', '200M', '250M', '300M', '350M'])





# Import df for a horizontal bar plot showing the gross of all 8 Star Wars movies

sw_gross = sw_budget_gross_plot[['movie_title_','gross']]

sw_gross_plot = sw_gross.sort_index(ascending=False)

ax= sw_gross_plot.plot(kind='barh', x='movie_title_', y='gross', color='gold')





# Set Title

ax.set_title('Star Wars Movies - Gross (in order of release date)')



# Set x-axis label

ax.set_xlabel("Gross US & CANADA (in millions)", labelpad=20, weight='bold', size=12)



# Set y-axis label

ax.set_ylabel("Movie Title", labelpad=20, weight='bold', size=12)



# Set xticks for gross

start, end = ax.get_xlim()

ax.xaxis.set_ticks(np.arange(start, end, 1000000000))

ax.set_xticks([0, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000,

              900000000, 1000000000])

ax.set_xticklabels(['0', '100M', '200M', '300M', '400M', '500M', '600M', '700M', '800M', '900M', '1B'])



plt.show()
import statistics

mean_budget_originals = [11000000.0, 18000000.0, 32500000.0]

mean_budget_prequels = [115000000.0, 115000000.0, 113000000.0]

mean_budget_sequels = [245000000.0, 317000000.0]



org_b = statistics.mean(mean_budget_originals) 

pre_b = statistics.mean(mean_budget_prequels)

seq_b = statistics.mean(mean_budget_sequels) 



print("Mean Budget of Episodes IV, V, & VI :", org_b)

print("Mean Budget of Episodes I, II, & III :", pre_b)

print("Mean Budget of Episodes VII & VIII :", seq_b)



mean_gross_originals = [460935665.0, 290158751.0, 309125409.0]

mean_gross_prequels = [474544677.0, 310675583.0, 380262555.0]

mean_gross_sequels = [936662225.0, 620181382.0]



org_g = statistics.mean(mean_gross_originals) 

pre_g = statistics.mean(mean_gross_prequels)

seq_g = statistics.mean(mean_gross_sequels) 



print("Gross mean of Episodes IV, V, & VI :", org_g)

print("Gross mean of Episodes I, II, & III :", pre_g)

print("Gross mean of Episodes VII & VIII", seq_g)



net_org = org_g - org_b

net_pre = pre_g - pre_b

net_seq = seq_g - seq_b



print("Mean Net Profit of Episodes IV, V, VI :", net_org)

print("Mean Net Profit of Episodes I, II, III :", net_pre)

print("Mean Net Profit of Episodes VII, VIII :", net_seq)
# Import df for a scatter plot displaying budget vs. gross of all 8 movies



ax=sw_budget_gross_plot.plot(kind='scatter', x='budget', y='gross', color='#2E80D0')

N=8



# Set labels

ax.set_xlabel('Budget (millions)', labelpad=20, weight='bold', size=12)

ax.set_ylabel('Gross US & CANADA (millions)', labelpad=20, weight='bold', size=12)

ax.set_title('Star Wars Movies I-VIII - Budget vs. Gross')



# Set xticks for budget

ax.set_xticks([0, 50000000, 100000000, 150000000, 200000000, 250000000, 300000000, 350000000])

ax.set_xticklabels(['0', '50M', '100M', '150M', '200M', '250M', '300M', '350M'])



# Set yticks for gross

ax.set_yticks([0, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 700000000, 800000000,

              900000000, 1000000000])

ax.set_yticklabels(['0', '100M', '200M', '300M', '400M', '500M', '600M', '700M', '800M', '900M', '1B'])



# Set legend

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



plt.show()
# Import df for a stacked bar chart displaying discrepancy between budget and gross of all 8 movies

ax=sw_arranged



# Identify the variables

Episode=['IV','V','VI','I','II', 'III', 'VII', 'VIII']

Budget_Gross=['budget','gross']

pos = np.arange(len(Episode))

y1= ax['budget']

y2= ax['gross']



# Plot a stacked barchart: bottom columns black for budget, top columns gold for gross

plt.bar(pos, y1, color='black', label='budget')

plt.bar(pos, y2, color='gold', bottom =y1, label = 'gross')



# Set xticks for episode number and y ticks for budget/gross values

plt.xticks(pos, Episode)

plt.yticks(np.arange(0, 1300000000, 100000000), ('0', '100M', '200M', '300M', '400M', '500M','600M', '700M', '800M', '900M', '1B', '1.1B', '1.2B', '1.3B'))

# Set xlabels, ylabels, title labels, and legend

plt.xlabel('Star Wars Episode', fontsize=16)

plt.ylabel('Budget & Gross (in millions)', fontsize=14)

plt.title('Budget vs. Gross of Star Wars Movies',fontsize=18)

plt.legend(Budget_Gross,loc=2)











plt.show()
ax = sw_arranged.plot(kind='line', x='movie_title_', y='imdb_score', marker='o', color='#EB214F')

N=8



ax.set_xlabel('Star Wars Episode ', labelpad=20, weight='bold', size=12)

ax.set_ylabel('IMDb Score (0-10)',  labelpad=20, weight='bold', size=12)

ax.set_title('Star Wars Movies IMDb Score')



plt.xticks(pos, Episode)

plt.yticks(np.arange(6, 10, 0.2), ('6.0', '6.2', '6.4', '6.6', '6.8', '7.0', '7.2', '7.4', '7.6', '7.8','8.0', '8.2', '8.4', '8.6', '8.8', '9.0', '9.2', '9.4', '9.6', '9.8', '10.0'))





ax = sw_arranged.plot(kind='line', x='movie_title_', y='gross', marker='o', color='#2E80D0')

N=8



ax.set_xlabel('Star Wars Episode ', labelpad=20, weight='bold', size=12)

ax.set_ylabel('Gross US & CANADA (in millions)',  labelpad=20, weight='bold', size=12)

ax.set_title('Star Wars Movies Gross')



plt.xticks(pos, Episode)

plt.yticks(np.arange(0, 1000000000, 100000000), ('0', '100', '200', '300', '400', '500', '600', '700', '800', '900', '1B'))
