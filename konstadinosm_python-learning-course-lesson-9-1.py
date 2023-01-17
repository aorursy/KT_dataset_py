# importing necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

%matplotlib inline
plt.rcParams['figure.figsize'] = [16.0, 16.0]

import os
print(os.listdir("../input"))
# importing dataset
df_original = pd.read_csv('../input/outbreaks.csv')
df_original.head(1)
missing_values = df_original.isnull().sum() * 100 / len(df_original)
missing_values
df = df_original.copy()

cols = [4,5,6,7,8]

df.drop(df.columns[cols], axis=1, inplace= True)
df.head(1)
df = df_original.copy()

my_cols = []

for k in range(0, len(df.columns)):
    if missing_values[k] > 30:
        my_cols.append(k)

df.drop(df.columns[my_cols], axis=1, inplace= True)
df.head(1)
df_state = df.filter(items=['State', 'Year', 'Illnesses', 'Fatalities']) # using .filter to alter the data perspective
df_state.head(3)
df_ill = df_state.filter(items=['State', 'Year', 'Illnesses']).copy()
df_fat = df_state.filter(items=['State', 'Year', 'Fatalities']).copy()
df_ill.head(3)
df_ill.sort_values(by='Illnesses', ascending=False)
df_ill = df_ill[df_ill.State != 'Multistate'] # getting rid of 'Multistate' entries
df_ill.sort_values(by='Illnesses', ascending=False).head(3)
df_ill_2006 = df_ill[df_ill.Year == 2006]
df_ill_2006.head(3)
table = pd.pivot_table(df_ill_2006, values=['Illnesses'], columns=['State'], aggfunc=np.sum)
table.head()
pd.pivot_table(df_ill_2006, values=['Illnesses'], columns=['State'], aggfunc=np.sum).plot.bar()
table.plot.bar()
df_fat = df_fat[df_fat.State != 'Multistate']
df_fat.head(3)
# total number of fatalities among all the years per state
df_fat.groupby(['State'])['Fatalities'].sum().sort_values(ascending= False).head(3)
df_state = df_state[df_state.State != 'Multistate']
df_state_summed = df_state.groupby(['State'])['Illnesses', 'Fatalities'].sum().sort_values(by=['Fatalities'], ascending= False).head(5).reset_index()
df_state_summed
print("State: " + str(df_state_summed.iloc[0][0]) + " (all years)")
print('')
print("Illnesses: " + str(df_state_summed.iloc[0][1]) + " (" + str(100 * df_state_summed.iloc[0][1] / (df_state_summed.iloc[0][1] + df_state_summed.iloc[0][2])) + "%)")
print("Fatalities: " + str(int(df_state_summed.iloc[0][2])) + " (" + str(100 * df_state_summed.iloc[0][2] / (df_state_summed.iloc[0][1] + df_state_summed.iloc[0][2])) + "%)")
print("Total Incidents: " + str(int(df_state_summed.iloc[0][1] + df_state_summed.iloc[0][2])))
print('')
df_state_summed.iloc[0][1:].plot.pie()
df_ill['encoded_state'] = None
df_ill.head(3)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_ill_enc = pd.DataFrame()
df_ill_enc['state'] = df_ill['State'].copy()
df_ill_enc.head(3)
df_ill_enc = df_ill_enc.apply(le.fit_transform)
df_ill['encoded_state'] = df_ill_enc['state']
df_ill.head(3)
df_state_encoding = df_ill.iloc[:,[0,-1]]
df_state_encoding = df_state_encoding.drop_duplicates(keep='first', inplace=False)
df_state_encoding.sort_values(by=['encoded_state'], ascending= True)
df_ill.plot.hexbin(x='Year', y='encoded_state', gridsize=50)
df_ill.plot.scatter(x='Year', y='encoded_state', s=df_ill['Illnesses'])
