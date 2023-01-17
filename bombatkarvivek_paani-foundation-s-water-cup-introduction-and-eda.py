import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pdf_StateLevelWinners = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv')

pdf_StateLevelWinners.T

# (21, 5)
pdf = pdf_StateLevelWinners[['District']].groupby('District')['District'].count().reset_index(name="count of vilages").sort_values('count of vilages', ascending=False)

# pdf.T

fig,ax = plt.subplots(1,2,figsize=(24,12))

sns.barplot(pdf['count of vilages'],pdf['District'],ax=ax[0])

pdf.set_index('District').plot.pie(y='count of vilages',legend=False,ax = ax[1])
pdf_ListOfTalukas = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv')

pdf_ListOfTalukas.T

# (184, 5)


pdf = pdf_ListOfTalukas[['Region']].groupby('Region')['Region'].count().reset_index(name="count of vilages").sort_values(by='count of vilages',ascending=False)

# pdf.T



fig, ax = plt.subplots(1,2,figsize=(24,12))

sns.barplot(pdf['count of vilages'],pdf['Region'],ax=ax[0])  #palette = 'Greens_d',

# plt.pie(pdf['count'],labels=pdf['District'])

pdf.set_index('Region').plot.pie(y='count of vilages', legend=False, ax=ax[1])
pdf_VillagesSupportedByDonations = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/VillagesSupportedByDonationsWaterCup2019.csv')

pdf_VillagesSupportedByDonations.T

# (332, 4)
pdf = pdf_VillagesSupportedByDonations[['District']].groupby('District')['District'].count().reset_index(name='count of vilages').sort_values('count of vilages',ascending=False)

# pdf

plt.subplots(figsize=(12,12))

sns.barplot(pdf['count of vilages'],pdf['District'])
pdf_MarkingSystem = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/MarkingSystem.csv')

pdf_MarkingSystem.T

# (332, 4)
fig, ax = plt.subplots(figsize=(12,12))

pdf_MarkingSystem.set_index('Component').plot.pie(y='Maximum Marks',legend=False, ax=ax)