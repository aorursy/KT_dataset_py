# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
init_df = pd.read_csv('../input/Kaggle v5 - Election Data.csv')

# Dropping Null Values
df = init_df.iloc[init_df.Party.dropna().index]

# Removing Dots in between, trailing or leading: These are often typos
df.Party = df.Party.str.replace('.','')
df.info()
# Checking for any discrepancies accross the dataset
df.Constituency_title.unique()
# We can see multiple names of NA-1
# Cleaning up extra spaces
df.Constituency_title = df.Constituency_title.str.strip()

# replacing empty spaces with a hyphen
df.Constituency_title = df.Constituency_title.str.replace(' ','-')

# converting values to upper case to avoid duplication
df.Constituency_title = df.Constituency_title.str.upper()
# Removing redundant axis created because of storing in CSV format
df = df.drop("Unnamed: 0",axis=1)
# Cleaning of the column: Seat
# Cleaning up extra spaces
df.Seat = df.Seat.str.strip()

# replacing empty spaces with a hyphen
df.Seat = df.Seat.str.replace(' ','-')

# converting values to upper case to avoid duplication
df.Seat = df.Seat.str.upper()
# Separating Seats to create district names
df['District'] = df.Seat.str.split('-').str[0]
df = df[['District', 'Seat', 'Constituency_title', 'Candidate_Name', 'Party', 'Votes',
       'Total_Valid_Votes', 'Total_Rejected_Votes', 'Total_Votes',
       'Total_Registered_Voters', 'Turnout', 'Election']]
# Standardizing Party Names of Some Major Parties in Pakistan
# Cleaning Spaces
df.Party = df.Party.str.strip()
df.Party = df.Party.str.upper()
# Standardizing party name "PTI"
parties = {"PTI":["PTI",'PAKISTAN TEHREEK-E-INSAF']}
indexes = df[df.Party.apply(lambda x: str(x).upper() in parties['PTI'])].index
print('Made ',len(indexes),' Changes')
df.loc[indexes,'Party'] = 'PTI'
# Standardizing party name "PPPP"
parties['PPPP'] = ['PAKISTAN PEOPLES PARTY PARLIAMENTARIANS', 'PPP', 'PPP-P']
indexes = df[df.Party.apply(lambda x: str(x).upper() in parties['PPPP'])].index
print('Made ',len(indexes),' Changes')
df.loc[indexes,'Party'] = 'PPPP'
# Standardizing party name "IND"

# Finding out how many instances have IND at the start
print(df[df.Party.str.startswith("IND")].Party.unique())

# Making Changes
parties['IND'] = ['IND', 'INDEPENDENT']
indexes = df[df.Party.apply(lambda x: str(x).upper() in parties['IND'])].index
print('Made ',len(indexes),' Changes')
df.loc[indexes,'Party'] = 'IND'
# Cleaning Typos in Party Names
df.loc[df[df.Party == ')ANP'].index,'Party'] = 'ANP'
df.loc[df[df.Party == '(KBG)'].index,'Party'] = 'KBG'
df.loc[df[df.Party == 'ALI PTI'].index,'Party'] = 'PTI'
df.loc[df[df.Party == 'JUI (HAZARVI)'].index,'Party'] = 'JUI (H)'
df.loc[df[df.Party == 'JUI(F)'].index,'Party'] = 'JUI (F)'
df.loc[df[df.Party == 'JUI(H)'].index,'Party'] = 'JUI (H)'
df.loc[df[df.Party == 'JAMIAT ULMA-E-ISLAM NAZRYATI PAKISTAN'].index,'Party'] = 'JUI (NAZRYATI)'
df.loc[df[df.Party == 'ISTIQLIL PARTY'].index,'Party'] = 'ISTIQLAL PARTY'
df.loc[df[df.Party == 'JAMIAT ULAMA-E-ISLAM (S)'].index,'Party'] = 'JAMIAT ULAMA-E-ISLAM PAKISTAN (S)'
df.loc[df[df.Party == 'JAMOTE QAMI MOVEMENT'].index,'Party'] = 'JAMOTE QAUMI MOVEMENT'
df.loc[df[df.Party == 'LABOYR PARTY PAKISTAN'].index,'Party'] = 'LABOUR PARTY PAKISTAN'
df.loc[df[df.Party == 'MUTTAHIDDA MAJLIS-E-AMAL PAKISTAN'].index,'Party'] = 'MUTTAHIDA MAJLIS-E-AMAL PAKISTAN'
df.loc[df[df.Party == 'MUTTHIDAÔØΩMAJLIS-E-AMALÔØΩPAKISTAN'].index,'Party'] = 'MUTTAHIDA MAJLIS-E-AMAL PAKISTAN'
df.loc[df[df.Party == 'PAKISTAN AMAN TEHREEK.'].index,'Party'] = 'PAKISTAN AMAN TEHREEK'
df.loc[df[df.Party == 'PAKISTAN MUSLIM LEAGUE ÔØΩHÔØΩ HAQIQI'].index,'Party'] = 'PAKISTAN MUSLIM LEAGUE HAQIQI'
df.loc[df[df.Party == 'PAKISTAN MUSLIM LEAGUE(J)'].index,'Party'] = 'PAKISTAN MUSLIM LEAGUE (J)'
df.loc[df[df.Party == 'PAKISTAN MUSLIM LEAGUE(QA)'].index,'Party'] = 'PAKISTAN MUSLIM LEAGUE (QA)'
df.loc[df[df.Party == 'PAKISTAN MUSLIM LEAGUE(Z)'].index,'Party'] = 'PAKISTAN MUSLIM LEAGUE (Z)'
df.loc[df[df.Party == 'PAKISTAN PEOPLES PARTY(SHAHEED BHUTTO)'].index,'Party'] = 'PAKISTAN PEOPLES PARTY (SHAHEED BHUTTO)'
df.loc[df[df.Party == 'PML (J) ¬©'].index,'Party'] = 'PML (J)'
df.loc[df[df.Party == 'PML (N'].index,'Party'] = 'PML (N)'
df.loc[df[df.Party == 'PML(Q)'].index,'Party'] = 'PML (Q)'
df.loc[df[df.Party == 'PML-N'].index,'Party'] = 'PML (N)'
df.loc[df[df.Party == 'PPIS'].index,'Party'] = 'PPI (SAG)'
# Getting All Party Names
party_indexes = df.Party.sort_values().index
init_df.loc[df.index,'Party'] = df.Party
party_names = init_df.iloc[party_indexes][['Party','Election']].drop_duplicates('Party')
print(party_names.to_string())
init_df.info()
# We can standardize more Parties in a similar way
init_df.to_csv('Kaggle v6 - Election Data.csv')
# Uploading New Version of the File
# Saving Party Names as a separate File
party_names.to_csv('party_names.csv')