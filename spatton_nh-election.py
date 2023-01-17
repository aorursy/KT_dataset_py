# NH Election Data

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Load data files

poll = pd.read_csv("/kaggle/input/new-hampshire-election-results/poll.csv")

elec = pd.read_csv("/kaggle/input/new-hampshire-election-results/elec.csv")

dist = pd.read_csv("/kaggle/input/new-hampshire-election-results/sr2004.csv")



print(poll.head())

print(dist.head())

print(elec.head())

# Answer some questions with the data to see how it fits together.

# What are the district designations for the polling station in Brentwood, NH?

print(poll[poll.town == 'Brentwood'])
# (Note that Brentwood is in Rockingham County House District 10)

# How many seats are in the house district for Brentwood?

print(dist[dist.office == 'SRROCK10'])
# Who ran for State Representative in Brentwood in 2010?

print(elec[(elec.year == 2010) & (elec.office == 'SRROCK10') ])
# What were the voting results for Auburn in 2008?

# We will print prettier with a few less columns

t_elec = elec[['year','office','party','candidate', 'town', 'count']]

print(t_elec[(t_elec.year == 2008) & (t_elec.town == 'Auburn') & (t_elec.office != 'CHECK')])



# NOTE: The offices are EC4 = Executive Council District 4

#                       GOV = Governor

#                       PRS = US President

#                       SRROCK03 = State Rep District 3 of Rockingham County

#                       SS14 = State Senate District 14

#                       UC1  = US Congress New Hampshire District 1

#                       US   = US Senate