import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# read in the csv file containing the data

kernels = pd.read_csv('../input/meta-kaggle/Kernels.csv')



# firstly let us remove our friend the Kaggle Kerneler, whose user Id is 2080166, from all the data:

kernels = kernels[kernels.AuthorUserId != 2080166]



# make a new column which is the ratio of views to votes for each notebook

kernels['views_per_vote'] = kernels['TotalViews']/kernels['TotalVotes']



# Some notebooks have 0 votes, and division by 0 results in inf: change these to a nan

kernels['views_per_vote'] = kernels['views_per_vote'].replace([np.inf, -np.inf], np.nan)

# now drop all the nan rows

kernels['views_per_vote'].dropna(inplace=True)



# finally make a histogram plot of this new column, 

# we shall arbitrarily set the cut off at 200 views per vote

plt.figure(figsize = (15,6))

ax = sns.distplot(kernels['views_per_vote'], 

             bins=200, 

             kde_kws={"clip":(0,200)}, 

             hist_kws={"range":(0,200)},

             color='darkcyan', 

             axlabel="Number of views per vote");

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
# first let us remove our friend the Kaggle Kerneler, whose user Id is 2080166, from the data:

#kernels = kernels[kernels.AuthorUserId != 2080166]



# convert the "MadePublicDate" column to a datetime

kernels['MadePublicDate'] = kernels['MadePublicDate'].apply(pd.to_datetime)

# make a 'days since' column for the number of days since going Public

date_today = pd.to_datetime('today')

kernels['days_since'] = (date_today - kernels['MadePublicDate']).dt.days

# now drop the kernels that are less than a 'month' (say 30 days) old

kernels.drop(kernels.index[kernels['days_since'] <= 30], inplace = True)

# now calculate the number of days that have passed between each vote for each kernel since going Public

kernels['days_per_vote'] = kernels['days_since']/kernels['TotalVotes']
# now plot a histogram (arbitrarily set the cut off at 1 year)

plt.figure(figsize = (15,6))

ax = sns.distplot(kernels['days_per_vote'], 

             bins=365,

             kde_kws={"clip":(1,365)}, 

             hist_kws={"range":(1,365)},

             color='darkcyan', 

             axlabel="Number of days per vote");

values = np.array([rec.get_height() for rec in ax.patches])

norm = plt.Normalize(values.min(), values.max())

colors = plt.cm.jet(norm(values))

for rec, col in zip(ax.patches, colors):

    rec.set_color(col)

plt.show();
total_number_of_kernels = kernels['days_per_vote'].shape[0]

kernels_that_take_month_per_vote =  kernels.index[kernels['days_per_vote'] >= 30].shape[0]

percentage = (100.0/total_number_of_kernels)*kernels_that_take_month_per_vote

print("percentage =",percentage)