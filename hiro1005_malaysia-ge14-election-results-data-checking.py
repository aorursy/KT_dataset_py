# Data file

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 

total = pd.read_csv('/kaggle/input/malaysia-ge14-election-results-parliament/Election-Results-2018 - Total_Votes_by_Seat.csv', header=0)

total
state = pd.read_csv('/kaggle/input/malaysia-ge14-election-results-parliament/Election-Results-2018 - State_Results_By_Candidate.csv', header=0)

state
parliament = pd.read_csv('/kaggle/input/malaysia-ge14-election-results-parliament/Election-Results-2018 - Parlimen_Results_By_Candidate.csv', header=0)

parliament
pd.set_option('display.max_rows', 2000)

plt.figure(figsize=(30, 40))

sns.countplot(x='State', data = total)
total['State'].value_counts()
senator_Number = total['State'].value_counts().sum()

print("Total number of Parliamentarians and State legislators : " + str(senator_Number))
population = pd.DataFrame({'State':['Sabah', 'Perak', 'Johor', 'Selangor', 'Kelantan', 'Pahang', 'Pulau Pinang', 'Kedah', 'Negeri Sembilan', 'Terengganu', 'Melaka', 'Sarawak', 'Perlis', 'Wilayah Persekutuan (KL)', 'Wilayah Persekutuan (Putrajaya)', 'Wilayah Persekutuan (Labuan)'], 'Seat':[85, 83, 82, 78, 59, 56, 53, 51, 44, 40, 34, 31, 18, 11, 1, 1], 'Population':[3903400, 2512100, 3764300, 6528400, 1885700, 1674600, 1774600, 2180600, 1130300, 1245700, 930700, 2812800, 254400, 1780700, 103800, 99300]})
population['Vote for Number to Win : '] = population['Population'] / population['Seat']



Average = population['Vote for Number to Win : '].mean()

print('Average number of votes required for one seat : ' + str(Average))
# Population divided by the average number of votes

population['The number of seats considered'] = population['Population'] // Average



# Number of seats considering the one-vote gap

population['The number of seats considered'] = population['The number of seats considered'] - population['Seat']

population

sort = population.sort_values('The number of seats considered', ascending=False)
plt.figure(figsize=(20, 20))

plt.rcParams["font.size"] = 10

left = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']

height = sort['The number of seats considered']

labels = sort['State']

plt.title("title")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.bar(left, height, width=0.9, color='#0096c8', edgecolor='b', linewidth=2, tick_label=labels)

plt.show()
# Gender of parliamentarians

sns.countplot(x='Gender', data =parliament)

parliament['Gender'].value_counts()

label = ['Male', 'Female']

plt.pie(parliament['Gender'].value_counts(), labels=label, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=0.7)

plt.show()
parliament['% of total Votes'] = parliament['% of total Votes'].str.strip('%.')

parliament['% of total Votes'] = parliament['% of total Votes'].astype(float)



pd.set_option('display.max_rows', 2000)

parliament.groupby("Seat Name")["% of total Votes"].sum()
parliament['Pekerjaan'].value_counts().head(10)
plt.figure(figsize=(20, 20))

plt.rcParams["font.size"] = 50

parliament['Coalition'].value_counts()

label = ['Harapan', 'BN', 'PAS', 'BEBAS'] 

plt.pie(parliament['Coalition'].value_counts(), labels=label, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=0.7)

plt.show()
parliament['Candidate Party'].value_counts()
plt.figure(figsize=(20, 20))

plt.rcParams["font.size"] = 50

parliament['Candidate Party'].value_counts()

label = ['BN', 'PKR', 'PAS', 'WARISAN', 'HR', 'DAP','BEBAS - KUNCI', 'PCS', 'PRM','SOLIDARITI', 'MU', 'BEBAS - BUKU', 'PSM', 'STAR', 'PPRS',  'BEBAS - GAJAH', 'PFP','PEACE', 

'ANAKNEGERI', 'BEBAS - RUMAH','SAPP', 'BEBAS - KAPAL TERBANG','BEBAS - KERETA', 'BEBAS - IKAN', 'AMANAH', 'BEBAS - KERUSI','BEBAS - TELEFON','BERJASA','PBDSB', 

'BEBAS - CAWAN',  'PBK','PAP', 'PCM'] 

plt.pie(parliament['Candidate Party'].value_counts(), labels=label, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=0.7)

plt.show()

#　Gender of State legislators

sns.countplot(x='Gender', data =state)

state['Gender'].value_counts()

label = ['Male', 'Female']

plt.pie(state['Gender'].value_counts(), labels=label, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=0.7)

plt.show()
state['Candidate Party'].value_counts()
plt.figure(figsize=(20, 20))

plt.rcParams["font.size"] = 50

state['Candidate Party'].value_counts()

label = ['BN', 'PKR', 'PAS', 'WARISAN', 'PRM', 'HR', 'MU', 'SOLIDARITI', 'BEBAS - KUNCI', 'PCS', 'PKS', 'PSM', 'PFP', 'ANAKNEGERI', 'PAP', 'PPRS', 'BEBAS - GAJAH', 'DAP', 'BEBAS - POKOK', 'SAPP', 'BEBAS - PEN', 'BEBAS - CAWAN', 'BEBAS - BUKU', 'BERJASA', 'STAR', 'USNO', 'BEBAS - KAPAL TERBANG', 'BEBAS - TRAKTOR', 'PCM', 'BEBAS - CINCIN', 'AMANAH', 'BEBAS - RUMAH', 'BEBAS - KUDA', 'BEBAS - IKAN', 'BEBAS - MOTOSIKAL', 'BEBAS - BEG', 'BEBAS - JAM', 'BEBAS - KERETA'] 

plt.pie(state['Candidate Party'].value_counts(), labels=label, counterclock=False, startangle=90, autopct='%1.1f%%', pctdistance=0.7)

plt.show()
state['% of total Votes'] = state['% of total Votes'].str.strip('%.')

state['% of total Votes'] = state['% of total Votes'].astype(float)

state.groupby("Seat Name")["% of total Votes"].sum()
state['Pekerjaan'].value_counts()