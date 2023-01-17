import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



f = '/kaggle/input/pga-tour-20102018-data/2019_data.csv'

data = pd.read_csv(f)
#Stats we are interested in

s1 = 'Driving Distance - (AVG.)'

s2 = 'Club Head Speed - (AVG.)'

s3 = 'Spin Rate - (AVG.)'

s4 = 'Hit Fairway Percentage - (%)'

s5 = 'FedExCup Season Points - (POINTS)'

stats = [s1,s2,s3,s4,s5]



#filter the dataset down to the 20 stats

df = data.loc[data['Variable'].isin(stats)]

df.head()
#Removing Commas

df['Value'] = df['Value'].map(lambda x:x.replace(',', ''))

#convert the 'Value' data to numeric values

df["Value"] = pd.to_numeric(df["Value"],errors='coerce')
#A pivot showing only Driving Distances per player

pivot = pd.pivot_table(data = df, index = 'Player Name', 

                       columns = data['Variable'][data['Variable']==s1], values = 'Value')



p = pd.DataFrame(pivot.agg([np.mean],axis=1))

p.describe()
pivot = pd.pivot_table(data = df, index = 'Player Name', columns = 'Variable', values = 'Value')

pivot.head()
sns.relplot(x=s2, y=s1,data=pivot)

sns.relplot(x=s2, y=s3,data=pivot)

sns.relplot(x=s3, y=s1,data=pivot)

plt.show()
sns.relplot(x=s1, y=s4,data=pivot)

sns.relplot(x=s1, y=s5,data=pivot)

sns.relplot(x=s4, y=s5,data=pivot)

plt.show()