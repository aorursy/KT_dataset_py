# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


data = pd.read_csv('../input/database.csv');

print(data.head());
#number of spill totals by state

spillsPerState = data['Accident State'].value_counts(normalize=True).sort_values(ascending=False);

spillsPerStateVol = data['Accident State'].value_counts();



#property damage totals by state

propertyDamage = data.groupby('Accident State')['Property Damage Costs'].sum().sort_values(ascending=False);



#fatality totals by state

fatalityTotals = data.groupby('Accident State')['All Fatalities'].sum().sort_values(ascending=False);



#'Environmental Remediation Costs' totals by state

EnvironRemedCosts = data.groupby('Accident State')['Environmental Remediation Costs'].sum().sort_values(ascending=False);



all = [spillsPerState, propertyDamage, fatalityTotals, EnvironRemedCosts];

bdf = pd.concat(all, axis=1).sort_values(by='Accident State', ascending=False)

print(bdf);


print(spillsPerState[spillsPerState > .02].sort_values())
plt.hist(spillsPerStateVol.drop('TX')); plt.show();
norm = lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x));

TexasDamages = data[data['Accident State'] == 'TX']['All Costs'].sort_values().dropna();

NTexasDamages = data[data['Accident State'] != 'TX']['All Costs'].dropna();

dmg = pd.DataFrame({'Texan Damage Costs': TexasDamages, 'Non-Texan Damage Costs' : NTexasDamages})

dmg.plot.hist(alpha=.5, color=["red", "blue"]);

plt.show();

print("Texan Damage costs:")

print(TexasDamages.describe())

print();

print("Non-Texan Damage costs:")

print(NTexasDamages.describe())

print();

print("Ratios, non-texan / texan")

print(NTexasDamages.describe()/TexasDamages.describe())
spillsByCompany = data['Operator Name'].value_counts();

spillsByCompanyNormed = data['Operator Name'].value_counts(normalize=True)

damagesByCompanyMeans = data.groupby('Operator Name')['All Costs'].mean().sort_values(ascending=False)

damagesByCompanyMeans.hist();

plt.show()

print("Raw # of spills by company");

print(spillsByCompany.head(10))

print()

print("Normed # of spills by company")

print(spillsByCompanyNormed.head(10))

print()

print("Mean costs by company")

print(damagesByCompanyMeans.head(10))
datn = data.dropna(subset=['Pipeline/Facility Name'])

datn[datn['Pipeline/Facility Name'].str.contains('KEY')]
transCanada = data[data['Operator Name'] == "TC OIL PIPELINE OPERATIONS INC"]['All Costs'].describe();

NtransCanada = data[data['Operator Name'] != "TC OIL PIPELINE OPERATIONS INC"]['All Costs'].describe();

print(transCanada)

print()

print(NtransCanada)

print()

print(transCanada/NtransCanada)
ysdm = data.groupby(['Accident State','Accident Year'])['All Costs'].mean();



quant = .95;

quantile = ysdm.quantile(q=quant);



print("For the ", quant, " quantile")



plt.hist(ysdm[ysdm < quantile]);

plt.show();



damageInQuantile = sum(ysdm[ysdm < quantile])

damageOutsideQuantile = sum(ysdm[ysdm > quantile])



print("Damages in the quantile", damageInQuantile);

print("Damages outside the quantile", damageOutsideQuantile);



print("Percentage of damage inside quantile ", damageInQuantile/(damageInQuantile + damageOutsideQuantile))



print(ysdm.describe())