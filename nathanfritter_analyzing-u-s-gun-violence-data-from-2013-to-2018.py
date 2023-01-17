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
# Read Data
gun_violence_data = '../input/gun-violence-data_01-2013_03-2018.csv'
gun_violence_df = pd.read_csv(gun_violence_data)
# Show head of data 
gun_violence_df.head()
missing_rows_df = gun_violence_df.isnull().sum() / gun_violence_df.shape[0]
print(missing_rows_df)
missing_rows_df.plot(kind = 'bar')
# Drop columns "location_description" and "participant_relationship" 
print(gun_violence_df.shape)
drop_columns = ['location_description', 'participant_relationship']
gun_violence_df = gun_violence_df.drop(drop_columns, axis = 1)
print(gun_violence_df.shape)
# Look at data types; what need converting?
# I.e. confirm district variables are decimals
gun_violence_df.info()
# Convert n_guns_involved to numeric
gun_violence_df['n_guns_involved_num'] = gun_violence_df['n_guns_involved'].fillna(-1).astype(int)
gun_violence_df.info()
# Convert district variables to object
district_variables = ['congressional_district', 'state_house_district', 'state_senate_district']
gun_violence_df[['congressional_district_obj', 'state_house_district_obj', 'state_senate_district_obj']] = gun_violence_df[['congressional_district', 'state_house_district', 'state_senate_district']].astype(object)
gun_violence_df.info()
# Drop original n_guns_involved column and distritct columns
gun_violence_df = gun_violence_df.drop(columns = ['congressional_district', 'state_house_district', 'state_senate_district', 'n_guns_involved'], axis = 1)
gun_violence_df.info()
# Import Natural Language Toolkit (nltk) package for text parsing and tokenizing
import nltk

# Subset data to test out our changes
subset = gun_violence_df.head(50)

# Columns to be split and parsed
parse_columns = ['gun_stolen', 'gun_type', 'incident_characteristics', 'participant_age', 'participant_age_group', 'participant_gender', 'participant_name', 'participant_status', 'participant_type']

# Make "incident_id" column the index
subset.set_index('incident_id')
# For empty strings: apply(lambda x: np.nan if isinstance(x,str) and x.isspace() and not x.str.len() > 0 else x)
subset['gun_stolen_parsed'] = subset['gun_stolen'].str.replace('\|\|', ', ')
subset['gun_stolen_parsed']

subset['gun_stolen_parsed'] =  subset['gun_stolen_parsed'].str.replace('::', ': ')
subset['gun_stolen_parsed']

col = subset['gun_stolen_parsed']
for index, item in col.iteritems():
    if isinstance(item, str):
        print(item)

#subset['gun_stolen_parsed_2'] = [word for sublist in subset['gun_stolen_parsed'] for word in sublist]
#subset['gun_stolen_parsed_2']
#subset['gun_stolen_parsed'] = subset['gun_stolen_parsed'].replace(',', '')
#subset['gun_stolen_parsed_2'] = subset['gun_stolen_parsed'].values.tolist().str.split(':')
#subset['gun_stolen_parsed_2']
#gun_violence_df_subset['gun_stolen_parsed'] = gun_violence_df_subset['gun_stolen'].replace(r'^\s+$', np.nan, regex = True).fillna('Missing Value').str.split('||')
#gun_violence_df_subset

# Describe Data
gun_violence_df.describe()
