import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/fifa19/data.csv")
data.corr()
f,ax = plt.subplots(figsize=(14,14))

sns.heatmap(data.corr(), linewidth=1)

print(data.columns)

print('\n')

print('Rows, Columns',data.shape)
data.head(5)
data.info()
data = data.drop(['Unnamed: 0','ID'],axis=1)

print(data.head(2))
print(data['Value'].head(5))
data['Value'] = data['Value'].str.replace('€','').str.replace('M','').str.replace('K','000')

data['Release Clause'] = data['Release Clause'].str.replace('€','').str.replace('M','').str.replace('K','000')
print(data['Value'],data['Release Clause'])
rounded_value = round(data['Value'].astype(float))

data['Value'] = rounded_value

for value in data['Value']:

    data.loc[data['Value'] < 1000, 'Value'] = data['Value'] * 1000000

    

print(data['Value'])

rounded_clause = round(data['Release Clause'].astype(float))

data['Release Clause'] = rounded_clause

for value in data['Release Clause']:

    data.loc[data['Release Clause'] < 1000, 'Release Clause'] = data['Release Clause'] * 1000000

    

print(data['Release Clause'])
data['Value'] = data['Value']/1000000

data['Release Clause'] = data['Release Clause']/1000000

print(data['Value'],data['Release Clause'])
data['delta_rating'] = data['Potential']-data['Overall']

data['price_per_point'] = data['Value']/data['Potential']

pot_data = data[data['Potential'] > 80]

top_data = pot_data.loc[:,['Name','Age','Nationality','Overall','Potential','Club','Value','Position','delta_rating','Release Clause','price_per_point']]

sort_data = top_data.sort_values(['Potential','delta_rating'],ascending = False)

print(sort_data.head(7))
sort_data['delta_rating'].value_counts(ascending=True)
filter_data = sort_data[sort_data['delta_rating']>10]

print(filter_data['Value'].value_counts(ascending=False))
gk = filter_data[filter_data['Position']== 'GK']



lb = filter_data[(filter_data['Position']== 'LB')|(filter_data['Position']=='RWB')]

rb = filter_data[(filter_data['Position']== 'RB')|(filter_data['Position']=='RWB')]

cb = filter_data[(filter_data['Position']== 'CB')|(filter_data['Position']=='LCB')|(filter_data['Position']=='RCB')]



cdm = filter_data[(filter_data['Position']== 'CDM')|(filter_data['Position']=='LDM')|(filter_data['Position']=='RDM')]

cam = filter_data[(filter_data['Position']== 'CAM')|(filter_data['Position']=='LAM')|(filter_data['Position']=='RAM')]

cm = filter_data[(filter_data['Position']== 'CM')|(filter_data['Position']=='LM')|(filter_data['Position']=='RM')]



lw = filter_data[(filter_data['Position']== 'LW')|(filter_data['Position']=='LF')]

rw = filter_data[(filter_data['Position']== 'RW')|(filter_data['Position']=='RF')]

st = filter_data[(filter_data['Position']== 'ST')|(filter_data['Position']=='CF')]
print(gk.head(5))
dream_team = pd.DataFrame()

dream_team[1544] = filter_data.loc[1544,:]

print(lb.head(5))
dream_team[8624] = filter_data.loc[8624,:]

print(cb.head(5))
dream_team[3776] = filter_data.loc[3776,:]

dream_team[7952] = filter_data.loc[7952,:]

print(rb.head(5))
dream_team[7525] = filter_data.loc[7525,:]

print(cdm.head(5))
dream_team[7414] = filter_data.loc[7414,:]

print(cm.head(5))
dream_team[6102] = filter_data.loc[6102,:]

print(cam.head(5))
dream_team[9485] = filter_data.loc[9485,:]

print(lw.head(5))
dream_team[5121] = filter_data.loc[5121,:]

print(st.head(5))
dream_team[7758] = filter_data.loc[7758,:]

print(rw.head(5))
dream_team[10087] = filter_data.loc[10087,:]

dream_team = dream_team.T

print(dream_team)
other_team = filter_data.loc[[1544,8624,3776,7952,7525,7414,6102,9485,5121,7758,10087],:]

print(other_team)
countries = filter_data['Nationality'].value_counts(ascending = False)

print(countries)