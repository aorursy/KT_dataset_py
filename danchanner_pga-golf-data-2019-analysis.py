import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

f = '/kaggle/input/pga-tour-20102018-data/2019_data.csv'

data = pd.read_csv(f)

data.head()
#Stats we are interested in



s1 = 'All-Around Ranking - (TOTAL)'

s2 = 'Average Approach Shot Distance - (AVG)'

s3 = 'Club Head Speed - (AVG.)'

s4 = 'Driving Distance - (AVG.)'

s5 = 'FedExCup Season Points - (POINTS)'

s6 = 'GIR Percentage from Fairway - (%)'

s7 = 'GIR Percentage from Other than Fairway - (%)'

s8 = 'Good Drive Percentage - (%)'

s9 = 'Greens in Regulation Percentage - (%)'

s10 = 'Hit Fairway Percentage - (%)'

s11 = 'Overall Putting Average - (AVG)'

s12 = 'Par 3 Scoring Average - (AVG)'

s13 = 'Par 4 Scoring Average - (AVG)'

s14 = 'Par 5 Scoring Average - (AVG)'

s15 = "Putting from - > 10' - (% MADE)"

s16 = "Putting from 4-8' - (% MADE)"

s17 = 'Putts Per Round - (AVG)'

s18 = 'Sand Save Percentage - (%)'

s19 = 'Scrambling - (%)'

s20 = 'Spin Rate - (AVG.)'



stats = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,

        s15,s16,s17,s18,s19,s20]

print(stats)

#filter the dataset down to the 20 stats

df = data.loc[data['Variable'].isin(stats)]

df.head()
#check for nulls

df.isna().sum()
#data.dropna(inplace=True)

#data.isna().sum()

df.info()
#convert the 'Value' data to numeric values

df["Value"] = pd.to_numeric(df["Value"],errors='coerce')

pivot = pd.pivot_table(data = df, index = 'Player Name', columns = 'Variable', values = 'Value')

pivot.head()
pivot.corr()
sns.relplot(x=s3, y=s4,size=s5, data=pivot)

plt.show()
sns.relplot(x=s4, y=s10,size=s5, data=pivot)

plt.show()
sns.relplot(x=s4, y=s5,hue=s9, data=pivot)

plt.show()
sns.relplot(x=s4, y=s13,size=s5, data=pivot)

sns.relplot(x=s4, y=s14,size=s5, data=pivot)

plt.show()
sns.relplot(x=s12, y=s5,size=s4, data=pivot)

sns.relplot(x=s13, y=s5, size=s4,data=pivot)

sns.relplot(x=s14, y=s5, size=s4,data=pivot)

plt.show()
sns.relplot(x=s9, y=s5,size=s4, data=pivot)

sns.relplot(x=s6, y=s5,size=s4, data=pivot)

sns.relplot(x=s7, y=s5,size=s4, data=pivot)

plt.show()
sns.relplot(x=s19, y=s5,hue=s16,data=pivot)

sns.relplot(x=s18, y=s5,hue=s16,data=pivot)

plt.show()
sns.relplot(x=s15, y=s5,hue=s9,size=s4,data=pivot)

sns.relplot(x=s16, y=s5,hue=s9,size=s4,data=pivot)

sns.relplot(x=s17, y=s5,hue=s9,size=s4,data=pivot)

plt.show()