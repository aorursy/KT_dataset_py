# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read in the data dictionary  

data_dictionary = pd.ExcelFile('/kaggle/input/us-supply-chain-information-for-covid19/cfs-2012-pum-file-users-guide-app-a-jun2015.xlsx')



# Print sheet names

#print(data_dictionary.sheet_names)

#['Data Dictionary', 'App A1', 'App A2', 'App A3', 'App A4']



#parse app1 and app2 

app1 = data_dictionary.parse('App A1', skiprows=[0,1], names=['MA', 'State','CFS_AREA','MA_TYPE','MA_Description'])

app2 = data_dictionary.parse('App A2', skiprows=[0], names=['NAICS', 'NAICS_Description'])



#read in app 3 and fill down file 

app3 = data_dictionary.parse('App A3', skiprows=[0], names=['Commodity_Code', 'Commodity_Description','Commodity_Group'])

app3['Commodity_Group'] = app3['Commodity_Group'].ffill() #need to fill down the page 

app3['Commodity_Code'] = app3['Commodity_Code'].apply(lambda x: str(x))



#app3 needs to be constructed from scratch 

# initialize list of lists 

data = [['02', 'Single Mode'], 

        ['03', 'Truck'],

        ['04', 'For-hire Truck'],

        ['05', 'Private Truck'],

        ['06', 'Rail'],

        ['07', 'Water'],

        ['08', 'Inland Water'],

        ['09', 'Great Lakes'],

        ['10', 'Deep Sea'],

        ['101', 'Multiple Waterways'],

        ['11', 'Air'],

        ['12', 'Pipeline'],

        ['13', 'Multiple Mode'],

        ['14', 'Parcel-USPS-Courier'],

        ['20', 'Non-parcel multimode'],

        ['15', 'Truck and Rail'],

        ['16', 'Truck and Water'],

        ['17', 'Rail and Water'],

        ['18', 'Other Multiple Mode'],

        ['09', 'Other Mode'],

        ['00', 'Mode Suppressed']] 

  

# Create the pandas DataFrame 

app4 = pd.DataFrame(data, columns = ['Mode_Code', 'Mode_Description'])

app3.head(20)







#read in the state names data for short hand reference if needed

states = pd.read_csv('/kaggle/input/us-supply-chain-information-for-covid19/state_code_to_name.csv')

states.head()
#read in the flat file for commodity flow survey 

cfs = pd.read_csv('/kaggle/input/us-supply-chain-information-for-covid19/cfs-2012-pumf-csv/cfs_2012_pumf_csv.txt')

cfs.head()
#reduce size of data base 

cfs=cfs[['ORIG_STATE','DEST_STATE','SCTG','SHIPMT_VALUE']]



cfs.head()
#cfs=cfs[(cfs['SCTG']=='21') | (cfs['SCTG']=='38')]

#cfs.shape
#get shape of cfs

#cfs_recode = pd.concat([cfs, states], axis=1, join='left')



#recode orig state 

cfs = pd.merge(cfs, states, how='left',left_on='ORIG_STATE', right_on='StateCode')

cfs = cfs.rename(columns = {"StateName":"ORIG_STATE_NAME"}) 

del cfs['StateCode']

del cfs['ORIG_STATE']



cfs.head()
#recode dest_state

cfs = pd.merge(cfs, states, how='left',left_on='DEST_STATE', right_on='StateCode')

cfs = cfs.rename(columns = {"StateName":"DEST_STATE_NAME"}) 

del cfs['StateCode']

del cfs['DEST_STATE']



cfs.head()
#recode sctg code 

cfs = pd.merge(cfs, app3, how='left',left_on='SCTG', right_on='Commodity_Code')

#cfs = cfs.rename(columns = {"MA_Description":"ORIG_MA_Description"}) 

del cfs['SCTG']

del cfs['Commodity_Code']

del cfs['Commodity_Group']



cfs.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

f, ax = plt.subplots(figsize=(15, 10))

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

sns.boxplot('Commodity_Description', 'SHIPMT_VALUE', data=cfs)

cfs=cfs[(cfs['Commodity_Description']=='Pharmaceutical Products') | (cfs['Commodity_Description']=='Precision Instruments and Apparatus')]

cfs.head()
sns.set()

f, ax = plt.subplots(figsize=(15, 10))

locs, labels = plt.xticks()

plt.setp(labels, rotation=90)

sns.boxplot('ORIG_STATE_NAME', 'SHIPMT_VALUE', data=cfs)
#cfs.pivot("ORIG_STATE_NAME","DEST_STATE_NAME","SHIPMT_VALUE")

cfs_nonzero = cfs[cfs.SHIPMT_VALUE>0]

cfs_state=cfs_nonzero.pivot_table(index='ORIG_STATE_NAME', 

                        columns='DEST_STATE_NAME', 

                        values='SHIPMT_VALUE')



f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(cfs_state, linewidths=1, ax=ax)