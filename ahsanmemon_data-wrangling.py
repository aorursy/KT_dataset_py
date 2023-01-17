import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output
import random

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(check_output(["ls", "../input"]).decode("utf8"))

import os
dataset_path = "../input/Kaggle v4 - Election Data.csv"
election_data = pd.read_csv(dataset_path)
# Cleaning the Additional Column "Unnamed: 0"
election_data = election_data.drop('Unnamed: 0',axis=1)
# Describing Dataset
election_data.describe()
# Finding total null values
election_data.isnull().sum()
# finding historical names of the Party PTI
election_data.Party.unique()
# Following Names could belong to PTI
# 'PTI',' PTI',' PTI ', 'PTI ', 'Ali PTI ', 'Pakistan Tehreek-e-Insaf'
# But they are showing separately majorly because of leading and trailing spaces

# Stripping Party names off of leading and trailing spaces
election_data['Party'] = election_data['Party'].str.strip()
election_data.Party.unique()
# Now we have 'PTI', 'Ali PTI 'and 'Pakistan Tehreek-e-Insaf'
# What is 'Ali PTI'?
print(election_data[election_data.Party == 'Ali PTI'])
# This guy was from PTI. The data entry says Ali PTI. Lets change it to PTI
index = election_data[election_data.Party == 'Ali PTI'].index
election_data.loc[index, 'Party'] = 'PTI'
# Now we are left with two names of PTI in the elections
# Lets create a dictionary for them
parties = {"PTI":["PTI",'Pakistan Tehreek-e-Insaf']}
# Similarly, Finding PPPPP Occurrences
# First we find out the mask, then we locate data with mask == True
mask = election_data.Party.dropna().str.contains("Pakistan Pe")
print(election_data.loc[mask[mask == True].index]['Party'].unique())
# Pakistan Peoples Party Parliamentatians is an alias of PPPP

mask = election_data.Party.dropna().str.contains("PPP")
print(election_data.loc[mask[mask == True].index]['Party'].unique())
# 'PPP', 'PPP-P' are also aliases of PPPP

# adding PPPP to dictionary and printing dictionary
parties['PPPP'] = ['Pakistan Peoples Party Parliamentarians', 'PPP', 'PPP-P']
print()
print(parties)
# Finding all PTI candidates
PTI_candidates = election_data[election_data['Party'].isin(parties['PTI'])]
PTI_candidates
# Finding all PPPP candidates
PPPP_candidates = election_data[election_data['Party'].isin(parties['PPPP'])]
PPPP_candidates
# finding distribution of PTI candidates over the past elections
print("\nPTI Candidate Distribution")
PTI_cand_dist = PTI_candidates.groupby("Election").count()['Seat']
print(PTI_cand_dist)

# finding distribution of PPPP candidates over the past elections
print("\nPPPP Candidate Distribution")
PPPP_cand_dist = PPPP_candidates.groupby("Election").count()['Seat']
print(PTI_cand_dist)

PPPP_cand_dist
# Using concat to combine both distributions 
party_dist = pd.concat([PTI_cand_dist,PPPP_cand_dist],axis=1).fillna(0)
party_dist.columns = ['PTI','PPPP']
party_dist
election_data.groupby(['Election'])['Candidate_Name'].count()
# Stripping constituency labels to avoid duplication
election_data['Constituency_title'] = election_data.Constituency_title.str.strip()

# Viewing Constituencies of 2002
elections_2002 = election_data[election_data['Election']==2008]
elections_2002_constituencies = list(set(election_data.Constituency_title))
print(pd.Series(elections_2002_constituencies))