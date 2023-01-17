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
df_2015 = pd.read_excel('/kaggle/input/2015_Final.xlsx')

df_2014 = pd.read_excel('/kaggle/input/2013_Final.xlsx')

df_2013 = pd.read_excel('/kaggle/input/2014_Final.xlsx')


lista = [df_2013,df_2014,df_2015]

#Function that collect the all 2013, 2014, 2015 election data and returns the major party and max votes received in that perticaul year 

#in respective pooling station

def Max_party_and_Votes():

        Max_party = pd.DataFrame()

        Max_party['Serial No. of Polling Station'] = df_2013['Serial No. of Polling Station']

        for i in range(3):

            Party = lista[i][['AAP','BJP', 'Congress']].idxmax(axis=1)

            max_vote= lista[i][['AAP','BJP', 'Congress']].max(axis=1)

            Party= pd.DataFrame(Party)

            max_vote = pd.DataFrame(max_vote)

            Party.columns.values[0] = str(i+13)

            max_vote.columns.values[0] = str(i+2013)

            Max_party = pd.concat([Max_party, Party,max_vote], 1)        

        return Max_party

#Function that collect the all 2013, 2014, 2015 election data and input as perticular party, 

#returns the given party votes in each year in separate column.  

def Party_votes_3yrs(x):

    party_all = pd.DataFrame()

    #party_all['Serial No. of Polling Station'] = df_2013['Serial No. of Polling Station']

    for i in range(3):

        party = lista[i][x]

        party= pd.DataFrame(party)

        party.columns.values[0] = str(x)+str(i+13)

        party_all = pd.concat([party_all, party], 1)    

    return party_all
#applying the function

Max_party =Max_party_and_Votes()

#changing the names of the columns

Max_party.rename(columns={13:'2013_Majority_Party', 14:'2014_Majority_Party', 15:'2015_Majority_Party',

                         2013:'2013_Max_vote',2014:'2014_Max_vote',2015:'2015_Maz_vote'}, inplace=True)

Max_party.head()
#applying the function, getting each party votes in all 3 yers in one dataframe

BJP =Party_votes_3yrs("BJP")

AAP= Party_votes_3yrs("AAP")

Congress = Party_votes_3yrs("Congress")

Final_3party = pd.concat([BJP, Congress, AAP], axis=1)

Final = pd.concat([Max_party,Final_3party], axis=1)

Final.head()
import matplotlib.pyplot as plt

import seaborn as sns

#performance of AAP in each polling station in the year 2013, 2014, 2015

plt.figure(figsize=(30,8))

sns.pointplot(x='Serial No. of Polling Station',y='AAP13',data=Final, color='red')

sns.pointplot(x="Serial No. of Polling Station", y="AAP14",data=Final, color='blue')

sns.pointplot(x="Serial No. of Polling Station", y="AAP15",data=Final, color='green')



# Observation- Greater margin increase in voting in 2015, among pooling stations (50-70 & 130-140)
#Year wise plot, how each party get votes compared to others 

plt.figure(figsize=(25,8))

sns.pointplot(x='Serial No. of Polling Station',y='BJP13',data=Final, color='orange')

sns.pointplot(x="Serial No. of Polling Station", y="AAP13",data=Final, color='blue')

sns.pointplot(x="Serial No. of Polling Station", y="Congress13",data=Final, color='green')

plt.figure(figsize=(25,8))

sns.pointplot(x='Serial No. of Polling Station',y='BJP14',data=Final, color='orange')

sns.pointplot(x="Serial No. of Polling Station", y="AAP14",data=Final, color='blue')

sns.pointplot(x="Serial No. of Polling Station", y="Congress14",data=Final, color='green')

plt.figure(figsize=(25,8))

sns.pointplot(x='Serial No. of Polling Station',y='BJP15',data=Final, color='orange')

sns.pointplot(x="Serial No. of Polling Station", y="AAP15",data=Final, color='blue')

sns.pointplot(x="Serial No. of Polling Station", y="Congress15",data=Final, color='green')



# Very important- year 2015- shows that in some of the pooling stations - BJP perfomed better ___ these are the very important for 

#AAP for campaining 
#number of pooling station got majority

plt.figure(figsize=(20,5))

plt.subplot(131)

sns.countplot(x='2013_Majority_Party', data=Final, color='red')

plt.subplot(132)

sns.countplot(x='2014_Majority_Party', data=Final, color='red')

plt.subplot(133)

sns.countplot(x='2015_Majority_Party', data=Final, color='red')
plt.figure(figsize=(20,5))

plt.subplot(131)

sns.stripplot(x = '2013_Majority_Party', y = "Serial No. of Polling Station", data =Final, jitter = True)

plt.subplot(132)

sns.stripplot(x = '2014_Majority_Party', y = "Serial No. of Polling Station", data =Final, jitter = True)

plt.subplot(133)

sns.stripplot(x = '2015_Majority_Party', y = "Serial No. of Polling Station", data =Final, jitter = True)
def label_race(row):

    if row['2013_Majority_Party']=='AAP' and row['2014_Majority_Party']=='AAP' and row['2015_Majority_Party']=='AAP':

        return 'AAP'

    if row['2013_Majority_Party']=='BJP' and row['2014_Majority_Party']=='BJP' and row['2015_Majority_Party']=='BJP':

        return 'BJP'

    if row['2013_Majority_Party']=='Congress' and row['2014_Majority_Party']=='Congress' and row['2015_Majority_Party']=='Congress' :

        return 'Congress'

    if row['2014_Majority_Party']=='BJP'and row['2015_Majority_Party']=='AAP' and (row['BJP14']-row['BJP15']<125):

        return row['2014_Majority_Party']



    else:

        return 'Other'

Final['Prediction_2020'] = Final.apply(label_race, axis=1)
Final
print(sum(Final['Prediction_2020']=="BJP"))

print(sum(Final['Prediction_2020']=="Other"))

print(sum(Final['Prediction_2020']=="AAP"))