import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

%config InlineBackend.figure_format='retina'

import seaborn as sns
data = pd.read_csv("../input/unions_states.csv")
fig, ax = plt.subplots(figsize=(6,6))

ax.plot(data.groupby('Year').Members.sum(), label='Members')

ax.plot(data.groupby('Year').Covered.sum(), label='Covered')

ax.ticklabel_format(axis='y', style='sci', scilimits=[-10,10])

ax.legend(['Total Members', 'Total Covered'], loc='best');

ax.set_title('Overall US Membership/Coverage in Unions 1983-2015', fontsize=12.5)

ax.set_yticklabels(['16', '18', '20', '22', '24', '26', '28'])

ax.set_ylabel('Total Persons (in Millions)', fontsize=11)

ax.set_xlabel('Year', fontsize=11);
fig1, ax = plt.subplots(figsize=(6,6))

ax.plot(data[data.Sector=='Private'].groupby('Year').Members.sum(), label='Members')

ax.plot(data[data.Sector=='Private'].groupby('Year').Covered.sum(), label='Covered')

ax.ticklabel_format(axis='y', style='sci', scilimits=[-10,10])

ax.set_yticklabels(['7', '8', '9', '10', '11', '12', '14'])

ax.set_ylabel("Total Persons (in millions)", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("US Private Sector Union Membership 1983-2015", fontsize=12.5)

ax.legend(['Members', 'Covered']);
fig2, ax = plt.subplots(figsize=(6,6))

ax.plot(data[data.Sector=='Public'].groupby('Year').Members.sum(), label='Members')

ax.plot(data[data.Sector=='Public'].groupby('Year').Covered.sum(), label='Covered')

ax.set_title("US Public Sector Union Membership 1983-2015", fontsize=13)

ax.ticklabel_format(axis="y", style="sci", scilimits=[-10,10])

ax.set_yticklabels(['5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9'])

ax.set_ylabel("Total Persons (in millions)", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.legend(['Members', 'Covered'], loc="lower right");
fig3, ax = plt.subplots(figsize=(6,6))

ax.plot(data[data.Sector=='Priv. Construction'].groupby('Year').Members.sum(), label='Members')

ax.plot(data[data.Sector=='Priv. Construction'].groupby('Year').Covered.sum(), label='Covered')

ax.set_title("US Private Construction Union Membership 1983-2015", fontsize=12.5)

ax.set_yticklabels(['80', '90', '100', '110', '120', '130', '140'])

ax.set_ylabel("Total Persons (in thousands)", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.legend(["Members", "Covered"], loc="best");
fig4, ax = plt.subplots(figsize=(6,6))

ax.plot(data[data.Sector=='Priv. Manufacturing'].groupby('Year').Members.sum(), label='Members')

ax.plot(data[data.Sector=='Priv. Manufacturing'].groupby('Year').Covered.sum(), label='Covered')

ax.set_title("US Manufacturing Union Membership 1983-2015", fontsize=12.5)

ax.set_yticklabels(['1', '2', '3', '4', '5', '6'])

ax.set_ylabel("Total Persons (in millions)", fontsize=11)

ax.set_xlabel('Year', fontsize=11)

ax.legend(["Members", "Covered"], loc="best");
#Create a table to get at year by year average overall membership between

#1983-2015

all_yr_sums=data.groupby('Year')[['Employment', 'Members', 'Covered']].sum()

all_yr_sums['Avg_Memb']=all_yr_sums.Members/all_yr_sums.Employment

all_yr_sums['Avg_Covd']=all_yr_sums.Covered/all_yr_sums.Employment
fig5, ax = plt.subplots(figsize=(6,6))

x=all_yr_sums.index

y=all_yr_sums[['Avg_Memb', 'Avg_Covd']]

ax.plot(x, y)

vals = ax.get_yticks()

ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

ax.set_ylabel("Membership level", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("Overall US Union membership as % of Economy", fontsize=12.5)

ax.legend(['Members', 'Covered'], loc="best");
#Private sector:

private = data[data.Sector=='Private']

private_sums=private.groupby('Year')[['Employment', 'Members', 'Covered']].sum()

private_sums['Avg_Memb']=private_sums.Members/private_sums.Employment

private_sums['Avg_Covd']=private_sums.Covered/private_sums.Employment
fig6, ax = plt.subplots(figsize=(6,6))

x=private_sums.index

y=private_sums[['Avg_Memb', 'Avg_Covd']]

ax.plot(x, y)

vals1 = ax.get_yticks()

ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals1])

ax.set_ylabel("Membership level", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("Private sector membership as % of Economy", fontsize=12.5)

ax.legend(['Members', 'Covered'], loc="best");
#Public sector:

public = data[data.Sector=='Public']

public_sums=public.groupby('Year')[['Employment', 'Members', 'Covered']].sum()

public_sums['Avg_Memb']=public_sums.Members/public_sums.Employment

public_sums['Avg_Covd']=public_sums.Covered/public_sums.Employment
fig7, ax = plt.subplots(figsize=(6,6))

x=public_sums.index

y=public_sums[['Avg_Memb', 'Avg_Covd']]

ax.plot(x, y)

vals2 = ax.get_yticks()

ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals2])

ax.set_ylabel("Membership level", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("Public sector membership as % of Economy", fontsize=12.5)

ax.legend(['Members', 'Covered'], loc="best");
#Construction sector:

construct = data[data.Sector=='Priv. Construction']

construct_sums=construct.groupby('Year')[['Employment', 'Members', 'Covered']].sum()

construct_sums['Avg_Memb']=construct_sums.Members/construct_sums.Employment

construct_sums['Avg_Covd']=construct_sums.Covered/construct_sums.Employment
fig8, ax = plt.subplots(figsize=(6,6))

x=construct_sums.index

y=construct_sums[['Avg_Memb', 'Avg_Covd']]

ax.plot(x, y)

vals3 = ax.get_yticks()

ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals3])

ax.set_ylabel("Membership level", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("Construction sector membership as % of Economy", fontsize=12.5)

ax.legend(['Members', 'Covered'], loc="best");
#Construction sector:

manu = data[data.Sector=='Priv. Manufacturing']

manu_sums=manu.groupby('Year')[['Employment', 'Members', 'Covered']].sum()

manu_sums['Avg_Memb']=manu_sums.Members/manu_sums.Employment

manu_sums['Avg_Covd']=manu_sums.Covered/manu_sums.Employment
fig9, ax = plt.subplots(figsize=(6,6))

x=manu_sums.index

y=manu_sums[['Avg_Memb', 'Avg_Covd']]

ax.plot(x, y)

vals4 = ax.get_yticks()

ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals4])

ax.set_ylabel("Membership level", fontsize=11)

ax.set_xlabel("Year", fontsize=11)

ax.set_title("Manufacturing sector membership as % of Economy", fontsize=12.5)

ax.legend(['Members', 'Covered'], loc="best");