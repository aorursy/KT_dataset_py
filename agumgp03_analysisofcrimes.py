import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

df = pd.read_csv('../input/crime-rates/report.csv')
df.head()
plt.rcdefaults()

fig,ax = plt.subplots(figsize=(20,20))



number_crimes = (df['violent_crimes']+df['homicides']+df['rapes']+df['assaults']+df['robberies'])/df['population']



ax.barh(df['agency_jurisdiction'], number_crimes, align='center')

ax.set_yticklabels(df['agency_jurisdiction'])

ax.set_xlabel('Total Number of Crimes')

ax.set_title('Crimes Around 40 Years')



plt.show()
per_year_homi = df.groupby('report_year').homicides.mean()

per_year_rapes = df.groupby('report_year').rapes.mean()

per_year_assaults = df.groupby('report_year').assaults.mean()

per_year_robberies = df.groupby('report_year').robberies.mean()
plt.figure(figsize=(15,8))

plt.title('Kind of Crime Around 40 Years', fontdict={'fontsize':18, 'fontweight':'bold'})



plt.plot(per_year_homi,'b-', label='Homicides')

plt.plot(per_year_rapes, 'g-', label='Rapes')

plt.plot(per_year_assaults, 'y-', label='Assaults')

plt.plot(per_year_robberies, 'r-', label='Robberies')



plt.yticks(np.arange(0,7000, step=300))



plt.xlabel('Year')

plt.ylabel('Total Number of Crimes')

plt.legend()

plt.show()
tot_homi = df.homicides.sum()

tot_rapes = df.rapes.sum()

tot_assaults = df.assaults.sum()

tot_robberies = df.robberies.sum()



plt.figure(figsize=(12,7))



label = ['Homicides','Rapes','Assaults','Robberies']

explode = [0.2,0.2,0,0]



plt.pie([tot_homi,tot_rapes,tot_assaults,tot_robberies], labels=label, autopct='%.2f %%', explode=explode)

plt.title('Crimes Around 40 Years', fontweight='bold')

plt.show()