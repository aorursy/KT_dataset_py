%pylab inline

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import re
prescribers = pd.read_csv('../input/prescriber-info.csv')

opioids = pd.read_csv('../input/opioids.csv')

ODs = pd.read_csv('../input/overdoses.csv')
import re

ops = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])

prescribed_ops = list(set(ops) & set(prescribers.columns))
prescribers['NumOpioids'] = prescribers.apply(lambda x: sum(x[prescribed_ops]),axis=1)

prescribers['NumPrescriptions'] = prescribers.apply(lambda x: sum(x.iloc[5:255]),axis=1)

prescribers['FracOp'] = prescribers.apply(lambda x: float(x['NumOpioids'])/x['NumPrescriptions'],axis=1)
prescribers.plot.scatter('NumOpioids','NumPrescriptions')
prescribers.hist('FracOp')
mean_NO = prescribers.groupby('Specialty')['NumOpioids'].mean().sort_values(ascending=False)

mean_fracO = prescribers.groupby('Specialty')['FracOp'].mean().sort_values(ascending=False)

mean_NO.head()
prescribers['O.Diff'] = prescribers.apply(lambda x: x['NumOpioids'] - mean_NO[x['Specialty']],axis=1)

prescribers['FracO.Diff'] = prescribers.apply(lambda x: x['FracOp'] - mean_fracO[x['Specialty']],axis=1)
prescribers
p = prescribers[prescribers['State'].isin(ODs['Abbrev'])]

op = p[p['Opioid.Prescriber']==1]['State'].value_counts()

tot_p = p['State'].value_counts()
ODs['Deaths'] = ODs['Deaths'].apply(lambda x: x.replace(',','')).astype('float')

ODs['Population'] = ODs['Population'].apply(lambda x: x.replace(',','')).astype('float')
ODs['DeathsPerCap'] = ODs['Deaths']/ODs['Population']*1E6

ODs['TotalPrescribers'] = ODs['Abbrev'].apply(lambda x: tot_p[x])

ODs['OPrescribers'] = ODs['Abbrev'].apply(lambda x: op[x])

ODs['FracOPrescribers'] = ODs['Abbrev'].apply(lambda x: op[x]/tot_p[x])
ODs.plot.scatter('FracOPrescribers','DeathsPerCap')
print("Correlation: %4f" % (ODs['FracOPrescribers'].corr(ODs['DeathsPerCap'])))