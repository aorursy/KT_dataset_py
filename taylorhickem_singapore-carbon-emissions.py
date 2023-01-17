# Using Python 3 environment 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import seaborn as sb
SQL_DB = '../input/carbon.db'

TABLES = {'energy':'energy_consumption','em_factors':'emission_factors'}

con = sqlite3.connect(SQL_DB)
em_factors = pd.read_sql('select * from ' + TABLES['em_factors'],con=con)

energy = pd.read_sql('select * from ' + TABLES['energy'],con=con)
em_factors
energy.head()
ghg = pd.merge(energy[[x for x in energy.columns if not x=='Fuel class']]

               ,em_factors,on='Fuel sub-class')
ghg['GtCO2eq'] = ghg.apply(lambda x: 

        x['Qty consumed']*x['TJ/unit']*x['tc / TJ']*x['CO2eq'],axis=1)
inv = pd.pivot_table(ghg,index=['Souce group','Source sub group'],columns='Fuel sub-class',values='GtCO2eq',aggfunc='sum')
inv
subtotals = inv.transpose().sum().reset_index()

subtotals.rename(columns={0:'GtCO2eq'},inplace=True)

subtotals
g = sb.barplot(x='Souce group',y='GtCO2eq',data=subtotals)

g.set_xticklabels(g.get_xticklabels(), rotation=90)